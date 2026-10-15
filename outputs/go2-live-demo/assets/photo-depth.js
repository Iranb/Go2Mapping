import * as THREE from 'three';
import {spatialManifest} from './spatial-photos.js?v=20261015-space';

const $ = id => document.getElementById(id);
const linear = x => x <= .04045 ? x/12.92 : ((x+.055)/1.055)**2.4;

function pointGeometry(buffer, colored=false) {
  const head = new DataView(buffer), count = head.getUint32(4, true), stride = colored ? 6 : 3;
  if (head.getUint32(0, true) !== (colored ? 3 : 1) || buffer.byteLength !== 8+count*stride*4) throw new Error('Invalid point cloud format');
  const data = new Float32Array(buffer, 8), positions = new Float32Array(count*3), colors = colored ? new Float32Array(count*3) : null;
  for (let i=0;i<count;i++) for (let k=0;k<3;k++) {
    positions[i*3+k] = data[i*stride+k];
    if (colored) colors[i*3+k] = linear(data[i*stride+3+k]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
  if (colors) geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
  geometry.computeBoundingSphere();
  return geometry;
}

async function fetchBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('Could not load photo 3D data');
  return response.arrayBuffer();
}

export class PhotoDepthView {
  constructor({command, toast, onReady, show}) {
    this.scene = new THREE.Scene();
    this.layer = new THREE.Group();
    this.scene.add(this.layer);
    this.grid = new THREE.GridHelper(12,24,0x3d6574,0x233747);
    this.grid.rotation.x=Math.PI/2;
    this.scene.add(this.grid);
    this.manifest=null;
    this.sequence=0;
    this.loadedId=null;
    this.loading=false;
    this.source=null;
    this.rgb=null;
    this.meshes=new THREE.Group();
    this.command=command;this.toast=toast;this.onReady=onReady;this.show=show;
    $('buildFusion').onclick=()=>this.build();
    for (const id of ['fusionLidar','fusionRGB','fusionSurface','fusionOpacity']) $(id).addEventListener('input',()=>this.visibility());
    $('depthFrame').onchange=()=>this.preview();
  }

  status(state) {
    this.state=state;
    $('buildFusion').disabled=state.active||state.fusion_busy||state.preview;
    $('buildFusion').textContent=state.fusion_busy?'Building photo 3D…':'Build / update photo 3D';
    $('buildFusion').disabled ||= !!state.spatial_capture;
    $('fusionStatus').textContent=state.fusion_message;
  }

  cameraModel() {
    return {...this.manifest?.camera,hfov_deg:Number($('cameraFov').value),pitch_deg:Number($('cameraPitch').value),yaw_deg:Number($('cameraYaw').value)};
  }

  async build() {
    $('buildFusion').disabled=true;
    try {
      await this.command('/api/fusion/build',{camera:this.cameraModel()});
      this.show();
    } catch(error) { this.toast(error.message); }
    finally { if(this.state)this.status(this.state); }
  }

  clear() {
    for(const object of [...this.layer.children]){
      object.traverse(child=>{child.geometry?.dispose();if(child.material){child.material.map?.dispose();child.material.dispose()}});
      this.layer.remove(object);
    }
    this.source=this.rgb=null;this.meshes=new THREE.Group();this.manifest=null;
  }

  async surface(base,frame) {
    if(!frame.triangles)return null;
    const buffer=await fetchBuffer(base+frame.surface), head=new DataView(buffer);
    const count=head.getUint32(4,true), indices=head.getUint32(8,true);
    if(head.getUint32(0,true)!==1||buffer.byteLength!==12+count*20+indices*4)throw new Error('Invalid textured surface format');
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(buffer,12,count*3),3));
    geometry.setAttribute('uv',new THREE.BufferAttribute(new Float32Array(buffer,12+count*12,count*2),2));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer,12+count*20,indices),1));
    geometry.computeBoundingSphere();
    try {
      const texture=await new THREE.TextureLoader().loadAsync(base+frame.image);
      texture.colorSpace=THREE.SRGBColorSpace;
      return new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,transparent:true,opacity:.9,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1}));
    } catch(error) { geometry.dispose();throw error; }
  }

  async load(manifest) {
    manifest=spatialManifest(manifest);
    if(manifest?.id===this.loadedId)return;
    const sequence=++this.sequence;
    this.loadedId=manifest?.id??null;
    this.loading=!!manifest;
    this.clear();
    $('fusionSummary').textContent=manifest?'Loading 3D data…':'No photo 3D map yet';
    $('depthFrame').replaceChildren();
    $('depthPair').hidden=true;
    $('depthPhoto').removeAttribute('src');$('depthImage').removeAttribute('src');
    $('depthCoverage').textContent='';$('fusionExport').removeAttribute('href');
    $('fusionExport').classList.add('hidden');
    if(!manifest){this.onReady(null);return}
    const base='/fusion/data/'+encodeURIComponent(manifest.id)+'/';
    let sourceGeometry=null, rgbGeometry=null;
    const meshes=[];
    try {
      sourceGeometry=pointGeometry(await fetchBuffer(base+manifest.lidar));
      rgbGeometry=pointGeometry(await fetchBuffer(base+manifest.colored),true);
      // Resolve every texture load before releasing a superseded dataset.
      const results=await Promise.allSettled(manifest.frames.map(frame=>this.surface(base,frame)));
      for(const result of results)if(result.status==='fulfilled'&&result.value)meshes.push(result.value);
      const failure=results.find(result=>result.status==='rejected');
      if(failure)throw failure.reason;
      if(sequence!==this.sequence){sourceGeometry.dispose();rgbGeometry.dispose();for(const mesh of meshes){mesh.geometry.dispose();mesh.material.map.dispose();mesh.material.dispose()}return}
      this.manifest=manifest;
      this.source=new THREE.Points(sourceGeometry,new THREE.PointsMaterial({color:0x91a5b5,size:.022,transparent:true,opacity:.32,depthWrite:false}));
      this.rgb=new THREE.Points(rgbGeometry,new THREE.PointsMaterial({vertexColors:true,size:.042,transparent:true,opacity:.98,depthWrite:false}));
      for(const mesh of meshes)this.meshes.add(mesh);
      this.layer.add(this.source,this.meshes,this.rgb);
      for(const station of manifest.stations??[]){
        const marker=new THREE.Mesh(new THREE.RingGeometry(.10,.14,24),new THREE.MeshBasicMaterial({color:0xffc578,side:THREE.DoubleSide,depthTest:false}));
        marker.position.set(station.position[0],station.position[1],manifest.floor+.03);
        marker.renderOrder=10;this.layer.add(marker);
      }
      this.grid.position.set(manifest.center[0],manifest.center[1],manifest.floor+.008);
      for(const [key,id]of [['hfov_deg','cameraFov'],['pitch_deg','cameraPitch'],['yaw_deg','cameraYaw']])$(id).value=manifest.camera[key];
      $('fusionSummary').textContent=(manifest.station_count??1)+' spatial stations · '+manifest.colored_points.toLocaleString()+' colored points · '+manifest.surface_triangles.toLocaleString()+' textured triangles';
      for(const frame of manifest.frames){const option=document.createElement('option');option.value=frame.slot;option.textContent='Station '+((frame.station??0)+1)+' · '+Math.round((frame.yaw*180/Math.PI+360)%360)+'°';$('depthFrame').append(option)}
      $('depthPair').hidden=false;
      $('fusionExport').href=base+manifest.export;$('fusionExport').classList.remove('hidden');
      $('fusionPairing').textContent=manifest.frames.some(f=>f.pairing==='historical-static-map')?'Legacy photos use the paired static lidar map; temporal pairing is approximate.':'Photos retain paired lidar and pose. Host receive times are used; there is no hardware synchronization.';
      this.preview();this.visibility();this.onReady(manifest);
    } catch(error) {
      sourceGeometry?.dispose();rgbGeometry?.dispose();for(const mesh of meshes){mesh.geometry.dispose();mesh.material.map?.dispose();mesh.material.dispose()}
      if(sequence===this.sequence){this.loadedId=null;$('fusionSummary').textContent='Load failed: '+error.message;this.toast(error.message)}
    } finally {if(sequence===this.sequence)this.loading=false;}
  }

  preview() {
    if(!this.manifest)return;
    const frame=this.manifest.frames.find(f=>f.slot===Number($('depthFrame').value));
    if(!frame)return;
    const base='/fusion/data/'+encodeURIComponent(this.manifest.id)+'/';
    $('depthPhoto').src=base+frame.image;$('depthImage').src=base+frame.depth_preview;
    $('depthCoverage').textContent='Measured lidar projection '+(frame.measured_pixels/(frame.depth_width*frame.depth_height)*100).toFixed(0)+'% · Voxel expansion and small-hole fill '+(frame.interpolated_pixels/(frame.depth_width*frame.depth_height)*100).toFixed(0)+'%';
  }

  visibility() {
    if(this.source)this.source.visible=$('fusionLidar').checked;
    if(this.rgb)this.rgb.visible=$('fusionRGB').checked;
    this.meshes.visible=$('fusionSurface').checked;
    const opacity=Number($('fusionOpacity').value);
    this.meshes.traverse(object=>{if(object.material){object.material.opacity=opacity;object.material.depthWrite=opacity>.95}});
  }
}

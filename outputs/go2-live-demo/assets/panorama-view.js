import * as THREE from 'three';
import {groupPhotos} from './spatial-photos.js?v=20261015-space';

const TAU=2*Math.PI;
const opticalToBody=new THREE.Matrix4().set(0,0,1,0,-1,0,0,0,0,-1,0,0,0,0,0,1);

export function sphericalPatch(left,right,bottom=-Math.PI/2,top=Math.PI/2){
  const columns=Math.max(2,Math.ceil((right-left)/.035)),rows=48,positions=[],uv=[],indices=[];
  for(let y=0;y<=rows;y++)for(let x=0;x<=columns;x++){
    const theta=left+(right-left)*x/columns,phi=bottom+(top-bottom)*y/rows,c=Math.cos(phi);
    positions.push(4*c*Math.cos(theta),4*c*Math.sin(theta),4*Math.sin(phi));uv.push(x/columns,y/rows);
  }
  for(let y=0;y<rows;y++)for(let x=0;x<columns;x++){
    const a=y*(columns+1)+x,b=a+1,c=a+columns+1,d=c+1;indices.push(a,b,c,b,d,c);
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));geometry.setIndex(indices);return geometry;
}

function camera(photo,texture){
  const [roll,pitch,yaw]=photo.rpy||[0,0,photo.yaw];
  const matrix=new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(roll,pitch,yaw,'ZYX')).multiply(opticalToBody);
  const w=texture.image.width,h=texture.image.height,f=w/(2*Math.tan(100*Math.PI/360));
  return {rotation:new THREE.Matrix3().setFromMatrix4(matrix).transpose(),intrinsics:new THREE.Vector4(f/w,f/h,(w-1)/(2*w),(h-1)/(2*h))};
}

function disposeMeshes(group){for(const mesh of [...group.children]){mesh.geometry.dispose();mesh.material.dispose();group.remove(mesh)}}

const vertex=`varying vec3 direction;varying float progress;void main(){direction=position;progress=uv.x;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;
const fragment=`
uniform sampler2D photoA;uniform sampler2D photoB;uniform mat3 cameraA;uniform mat3 cameraB;
uniform vec4 lensA;uniform vec4 lensB;varying vec3 direction;varying float progress;
vec4 samplePhoto(sampler2D photo,mat3 camera,vec4 lens){
 vec3 optical=camera*normalize(direction);if(optical.z<=0.)return vec4(0.);
 vec2 uv=optical.xy/optical.z*lens.xy+lens.zw;uv.y=1.-uv.y;
 if(uv.x<0.||uv.x>1.||uv.y<0.||uv.y>1.)return vec4(0.);
 return vec4(texture2D(photo,uv).rgb,1.);
}
void main(){
 vec4 a=samplePhoto(photoA,cameraA,lensA),b=samplePhoto(photoB,cameraB,lensB);
 if(a.a+b.a==0.)discard;float blend=smoothstep(.3,.7,progress);
 float wa=(1.-blend)*a.a,wb=blend*b.a;
 if(wa+wb<.0001){wa=a.a;wb=b.a;}
 gl_FragColor=vec4((a.rgb*wa+b.rgb*wb)/max(.0001,wa+wb),1.);
 #include <colorspace_fragment>
}`;

export class PanoramaView{
  constructor(scene,onReady){this.scene=scene;this.onReady=onReady;this.frames=new Map();this.sequence=0;this.fusedSequence=0;this.loadedId=null;this.fusedTexture=null;this.blended=true;this.preview=new THREE.Group();this.fused=new THREE.Group();scene.add(this.preview,this.fused)}
  get count(){return this.photos?.length??0}
  get ready(){return !!this.fusedTexture||this.frames.size>0}

  async update(photos){
    this.photos=photos;this.stations=groupPhotos(photos);
    const id=this.stations.some(s=>s.id===this.stationId)?this.stationId:this.stations[0]?.id??null;
    return this.setStation(id);
  }

  async setStation(id){
    const station=this.stations?.find(s=>s.id===id);
    const changed=this.stationId!==(station?.id??null);
    this.stationId=station?.id??null;
    if(changed){
      // Clear the previous location before moving the panorama's camera centre.
      for(const frame of this.frames.values())frame.texture.dispose();
      this.frames.clear();disposeMeshes(this.preview);this.loadView(null);
    }
    if(station)this.setOrigin(new THREE.Vector3(station.position[0],station.position[1],station.position[2]+.04));
    const photos=station?.photos??[];
    const sequence=++this.sequence,created=[];
    const frames=await Promise.all(photos.map(async photo=>{
      const old=this.frames.get(photo.slot);if(old?.photo.version===photo.version)return old;
      const texture=await new THREE.TextureLoader().loadAsync('/photos/'+photo.slot+'.jpg?v='+photo.version).catch(()=>null);
      if(!texture)return null;texture.colorSpace=THREE.SRGBColorSpace;texture.minFilter=THREE.LinearFilter;texture.generateMipmaps=false;
      const frame={photo,texture};created.push(frame);return frame;
    }));
    if(sequence!==this.sequence){for(const frame of created)frame.texture.dispose();return}
    const valid=frames.filter(Boolean),retained=new Set(valid.map(frame=>frame.texture));
    for(const old of this.frames.values())if(!retained.has(old.texture))old.texture.dispose();
    this.frames=new Map(valid.map(frame=>[frame.photo.slot,frame]));this.rebuild();this.onReady();
    await this.loadView(this.stationManifest());
  }

  rebuild(){
    disposeMeshes(this.preview);
    const frames=[...this.frames.values()].sort((a,b)=>(a.photo.yaw%TAU+TAU)%TAU-(b.photo.yaw%TAU+TAU)%TAU);
    for(let i=0;i<frames.length;i++){
      const a=frames[i],b=frames[(i+1)%frames.length],left=(a.photo.yaw%TAU+TAU)%TAU;
      const right=left+((b.photo.yaw-a.photo.yaw+TAU)%TAU||TAU),ca=camera(a.photo,a.texture),cb=camera(b.photo,b.texture);
      const material=new THREE.ShaderMaterial({uniforms:{photoA:{value:a.texture},photoB:{value:b.texture},cameraA:{value:ca.rotation},cameraB:{value:cb.rotation},lensA:{value:ca.intrinsics},lensB:{value:cb.intrinsics}},vertexShader:vertex,fragmentShader:fragment,side:THREE.DoubleSide,depthWrite:false});
      this.preview.add(new THREE.Mesh(sphericalPatch(left,right),material));
    }
    this.setBlended(this.blended);
  }

  setOrigin(origin){this.preview.position.copy(origin);this.fused.position.copy(origin)}
  setBlended(value){this.blended=value;this.preview.visible=!this.fusedTexture||!value;this.fused.visible=!!this.fusedTexture&&value}

  stationManifest(){
    const manifest=this.spatialManifest;
    if(!manifest)return null;
    const station=manifest.stations?.find(s=>s.station===this.stationId);
    if(station)return {...station,id:manifest.id+':'+station.station};
    return (this.stations?.length??0)<=1?manifest:null;
  }

  async load(manifest){this.spatialManifest=manifest;return this.loadView(this.stationManifest())}

  async loadView(manifest){
    if((manifest?.id??null)===this.loadedId)return;
    const sequence=++this.fusedSequence;this.loadedId=manifest?.id??null;
    this.fusedTexture?.dispose();this.fusedTexture=null;disposeMeshes(this.fused);this.preview.visible=true;this.onReady();
    if(!manifest)return;
    const texture=await new THREE.TextureLoader().loadAsync(manifest.image).catch(()=>null);
    if(!texture){if(sequence===this.fusedSequence)this.loadedId=null;return}
    if(sequence!==this.fusedSequence){texture.dispose();return}
    texture.colorSpace=THREE.SRGBColorSpace;texture.minFilter=THREE.LinearFilter;texture.generateMipmaps=false;
    texture.wrapS=THREE.RepeatWrapping;this.fusedTexture=texture;
    this.fused.add(new THREE.Mesh(sphericalPatch(0,TAU,manifest.latitude_min,manifest.latitude_max),new THREE.MeshBasicMaterial({map:texture,transparent:true,side:THREE.DoubleSide,depthWrite:false})));
    this.setBlended(this.blended);this.onReady();
  }
}

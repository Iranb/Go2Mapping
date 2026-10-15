import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const root=new URL('../outputs/go2-live-demo/',import.meta.url);
const realThree=await import('data:text/javascript;base64,'+readFileSync(new URL('vendor/three.module.js',root)).toString('base64'));
const grouping=await import('data:text/javascript;base64,'+readFileSync(new URL('assets/spatial-photos.js',root)).toString('base64'));
const photo=(slot,x,epoch=1)=>({slot,version:1,position:[x,0,.3],yaw:slot*.2,rpy:[0,0,slot*.2],connection_epoch:epoch});
assert.deepEqual(grouping.groupPhotos([photo(0,0),photo(1,.3),photo(2,.6),photo(3,0,2)]).map(s=>s.photos.length),[2,1,1]);
const legacy=grouping.spatialManifest({id:'legacy',frames:[photo(0,0),photo(1,1)]});
assert.equal(legacy.station_count,2);assert.deepEqual(legacy.frames.map(f=>f.station),[0,1]);
assert.equal(grouping.spatialManifest(legacy),legacy);

const pending=[];
class Loader{
  loadAsync(url){return new Promise(resolve=>pending.push({url,resolve}))}
}
const ctx={THREE:{...realThree,TextureLoader:Loader},groupPhotos:grouping.groupPhotos};
vm.createContext(ctx);
vm.runInContext(readFileSync(new URL('assets/panorama-view.js',root),'utf8').replace(/^import.*\n/gm,'').replaceAll('export ',''),ctx);
const PanoramaView=vm.runInContext('PanoramaView',ctx);
const view=new PanoramaView(new realThree.Scene(),()=>{});
const tick=async()=>{for(let i=0;i<6;i++)await Promise.resolve()};
function resolveImage(part){
  const i=pending.findIndex(item=>item.url.includes(part));assert(i>=0,'Expected pending '+part);
  const {resolve}=pending.splice(i,1)[0];
  const texture=new realThree.Texture({width:960,height:540});texture.disposals=0;
  texture.addEventListener('dispose',()=>texture.disposals++);resolve(texture);return texture;
}
let request=view.update([photo(0,0),photo(1,1)]);
assert.deepEqual(pending.map(p=>p.url),['/photos/0.jpg?v=1'],'Only the selected position loads textures');
const first=resolveImage('/photos/0');await request;
assert.equal(view.frames.size,1);assert.equal(view.count,2);
const manifest={id:'one',stations:[0,1].map(station=>({station,image:'/panorama-'+station,latitude_min:-1,latitude_max:1}))};
request=view.load(manifest);resolveImage('/panorama-0');await request;
assert(view.fusedTexture);

request=view.setStation(1);
assert.equal(view.frames.size,0);assert.equal(view.fusedTexture,null);
assert.equal(view.preview.children.length,0,'The previous photo must not flash at a new position');
assert.equal(view.preview.position.x,1);assert.equal(first.disposals,1);
resolveImage('/photos/1');await tick();resolveImage('/panorama-1');await request;
assert.deepEqual([...view.frames.keys()],[1]);

const old=view.setStation(0);const newest=view.setStation(1);
const stalePhoto=resolveImage('/photos/0');await old;
assert.equal(stalePhoto.disposals,1);assert.equal(view.frames.size,0);
resolveImage('/photos/1');await tick();resolveImage('/panorama-1');await newest;
assert.deepEqual([...view.frames.keys()],[1]);

const staleFusionRequest=view.load({...manifest,id:'two'});
const move=view.setStation(0);
const staleFusion=resolveImage('/panorama-1');await staleFusionRequest;
assert.equal(staleFusion.disposals,1);assert.equal(view.fusedTexture,null);
resolveImage('/photos/0');await tick();const currentFusion=resolveImage('/panorama-0');await move;
assert.equal(view.fusedTexture,currentFusion);assert.equal(view.loadedId,'two:0');
assert.equal(pending.length,0);

// Execute the actual page's browsing function with a command trap.
const html=readFileSync(new URL('index.html',root),'utf8');
const selection=html.slice(html.indexOf('async function selectStation('),html.indexOf('function stepStation('));
const selected=[];let resolveOld;
const browser={view:'photo',panoYaw:0,panoPitch:1,spatialStations:()=>[{id:0,photos:[{yaw:1}]},{id:1,photos:[{yaw:2}]}],panorama:{stationId:0,setStation(id){this.stationId=id;return id===0?new Promise(r=>resolveOld=r):Promise.resolve()}},placePhotoCamera(){selected.push(this?.panorama?.stationId)},updateStations(){},command(){throw Error('Browsing must not send commands')}};
vm.createContext(browser);vm.runInContext(selection,browser);
const older=browser.selectStation(0);await browser.selectStation(1);resolveOld();await older;
assert.equal(browser.panoYaw,2,'A stale selection must not overwrite the newest camera heading');
browser.view='fusion';browser.spatialStations=()=>[{id:0,position:[3,4,.3]}];
const poses=[];browser.photoDepth={manifest:{frames:[{station:0,yaw:0}]}};
browser.camera={position:{set(...p){poses.push(p)}}};browser.controls={target:{set(){}},update(){}};
await browser.selectStation(0);assert.deepEqual(poses,[[3,4,.33999999999999997]]);
const capture=html.slice(html.indexOf('function updateSpatialCapture('),html.indexOf('function updateStatus('));
const elements=new Map();browser.clientId='operator';browser.$=id=>{if(!elements.has(id))elements.set(id,{});return elements.get(id)};
vm.runInContext(capture,browser);
const status={connected:true,pose:{position:[0,0,.3]},state_age_s:.1,lidar_age_s:.1,spatial_capture:false,active:false,fusion_busy:false,spatial_message:'ready'};
browser.updateSpatialCapture(status);assert.equal(elements.get('spatialCapture').disabled,false);
browser.updateSpatialCapture({...status,lidar_age_s:null});assert.equal(elements.get('spatialCapture').disabled,true);
assert.match(elements.get('spatialStatus').textContent,/Waiting for live lidar/);
browser.updateSpatialCapture({...status,lidar_age_s:800,spatial_capture:true});assert.equal(elements.get('spatialCapture').disabled,false,'Stale sensors must not prevent ending capture');
console.log('8 spatial UI scenarios passed: anchored/epoch grouping, legacy dataset positions, lazy loading, immediate old-view cleanup, stale photo rejection, stale panorama rejection, camera-only browsing, and sensor-readiness controls.');

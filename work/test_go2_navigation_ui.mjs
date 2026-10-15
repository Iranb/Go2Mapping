import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const source=readFileSync(new URL('../outputs/go2-live-demo/index.html', import.meta.url),'utf8');
const fn=source.slice(source.indexOf('async function planTarget('),source.indexOf("$('replan').onclick="));
for(const change of ['none','scene','epoch','owner','newer']){
 let resolve, calls=[];
 const ctx={planning:false,clientId:'test',lastState:{active:false,scene_generation:1,connection_epoch:2,navigation:null},setSelecting(){},updateNavigation(){},toast(){},command(path,body){calls.push([path,body]);return new Promise(r=>resolve=r)}};
 vm.createContext(ctx);vm.runInContext(fn,ctx);
 const pending=ctx.planTarget([1,2]);
 assert.equal(calls.length,1);assert.equal(calls[0][0],'/api/plan');
 await ctx.planTarget([3,4]);assert.equal(calls.length,1,'Repeated clicks must not queue');
 if(change==='scene')ctx.lastState.scene_generation=3;
 if(change==='epoch')ctx.lastState.connection_epoch=3;
 if(change==='owner')ctx.clientId='other';
 if(change==='newer')ctx.lastState.navigation={id:9};
 resolve({id:8,target:[1,2],reachable:true});await pending;
 assert.equal(ctx.planning,false);
 if(change==='none')assert.equal(ctx.lastState.navigation.id,8);
 else if(change==='newer')assert.equal(ctx.lastState.navigation.id,9);
 else assert.equal(ctx.lastState.navigation,null,'Stale responses must not restore old targets');
 assert(calls.every(([path])=>path==='/api/plan'),'Planning must never execute navigation');
}
console.log('5 UI planning scenarios passed: normal, scene reset, reconnect, operator change, newer plan; repeated clicks do not queue or execute motion.');
const renderFn=source.slice(source.indexOf('function updateNavigation(s){'),source.indexOf("let view='map'"));
const geometry=()=>({dispose(){},setFromPoints(){return this}});
const marker=()=>({visible:false,position:{set(){}},geometry:geometry(),computeLineDistances(){}});
const elements=new Map();
const ctx={clientId:'test',selecting:false,planning:false,goPending:false,navSignature:'',navActionSignature:'',setSelecting(){},$:id=>{if(!elements.has(id))elements.set(id,{textContent:'',disabled:false,hidden:false,replaceChildren(){},append(){}});return elements.get(id)},document:{createElement(){return {}}},goalMarker:marker(),goalMaterial:{color:{setHex(){}}},goalRing:{geometry:geometry()},route:marker(),blockMarker:marker(),blockMaterial:{color:{setHex(){}}},landingMarker:marker(),remainingRoute:marker(),THREE:{RingGeometry:class{dispose(){}},BufferGeometry:class{dispose(){}setFromPoints(){return this}},Vector3:class{}}};
vm.createContext(ctx);vm.runInContext(renderFn,ctx);
const base={connection_epoch:2,scene_generation:1,active:false,nav_ready:true,fusion_busy:false,floor:0,pose:{position:[0,0,.3],yaw:0},navigation:{id:1,owner:'test',epoch:2,scene_generation:1,start:[0,0],yaw:0,target:[1,0],path:[[0,0],[.5,0]],floor:0,reachable:true,used:false,expired:false,length_m:.5,walk_count:1,turn_count:0,arrival_m:.2,partial:true,goal_offset_m:.5,arrival_target:[.5,0],state:'planned'}};
for(const scenario of ['partial','complete','expired','active','moved','owner','epoch','scene','cleared','blocked_expired']){
 const s=structuredClone(base);
 if(scenario==='complete')s.navigation.partial=false;
 if(scenario==='expired')s.navigation.expired=true;
 if(scenario==='active')s.active=true;
 if(scenario==='moved')s.pose.position[0]=.12;
 if(scenario==='owner')s.navigation.owner='other';
 if(scenario==='epoch')s.connection_epoch++;
 if(scenario==='scene')s.scene_generation++;
 if(scenario==='cleared')s.navigation=null;
 if(scenario==='blocked_expired')Object.assign(s.navigation,{reachable:false,expired:true,reason:'Select a target within 4 m of the robot'});
 ctx.updateNavigation(s);
 const go=elements.get('go'),replan=elements.get('replan');
 assert.equal(go.disabled,!['partial','complete'].includes(scenario),scenario+' execute availability');
 assert.equal(replan.disabled,['active','epoch','scene','cleared'].includes(scenario),scenario+' replan availability');
 if(scenario==='blocked_expired')assert.equal(elements.get('navstate').textContent,s.navigation.reason);
 if(scenario==='partial'){assert.equal(ctx.landingMarker.visible,true);assert.equal(ctx.remainingRoute.visible,true);assert.equal(go.textContent,'Go to temporary stop')}
 if(scenario==='complete'||scenario==='cleared'){assert.equal(ctx.landingMarker.visible,false);assert.equal(ctx.remainingRoute.visible,false)}
}
console.log('10 UI rendering scenarios passed: partial/complete markers, expiry, motion, pose change, operator ownership, reconnect, scene change, clear, and blocked-plan explanations.');

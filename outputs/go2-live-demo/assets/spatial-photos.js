// Match the server's anchored 35 cm grouping; distant photos never share a sphere.
export function groupPhotos(photos){
  const groups=[];
  for(const photo of [...photos].sort((a,b)=>a.slot-b.slot)){
    const near=groups.filter(s=>s.epoch===photo.connection_epoch&&Math.hypot(...s.position.map((v,i)=>v-photo.position[i]))<=.35);
    near.sort((a,b)=>Math.hypot(...a.position.map((v,i)=>v-photo.position[i]))-Math.hypot(...b.position.map((v,i)=>v-photo.position[i])));
    let station=near[0];
    if(!station){station={id:groups.length,position:[...photo.position],epoch:photo.connection_epoch,photos:[]};groups.push(station)}
    station.photos.push(photo);
  }
  return groups;
}

export function spatialManifest(manifest){
  if(!manifest||manifest.stations)return manifest;
  const groups=groupPhotos(manifest.frames),bySlot=new Map();
  const stations=groups.map(s=>{for(const p of s.photos)bySlot.set(p.slot,s.id);return {id:s.id,position:s.position,slots:s.photos.map(p=>p.slot)}});
  return {...manifest,stations,station_count:stations.length,spatial:stations.length>1,
    frames:manifest.frames.map(f=>({...f,station:bySlot.get(f.slot)}))};
}

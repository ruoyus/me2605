// Reproducible classroom presets. Prints measured results, not expected rankings.
const E=require('./engine.js');
const presets={control:{depth:4,batch:32,scale:1},scale:{depth:8,batch:32,scale:100},batch:{depth:4,batch:2,scale:1}};
for(const seed of [7,11,23])for(const [name,p] of Object.entries(presets)){
 const c={width:16,lr:.03,steps:300,placement:'post',affine:true,seed,...p};
 const train=E.moons(seed,768,c.scale),test=E.moons(seed+100,256,c.scale);
 const models=['none','ln','bn'].map(arm=>({arm,net:E.create(c,arm),failed:false}));
 for(let s=1;s<=c.steps;s++){
  const idx=E.indices(seed,s,c.batch,train.n);
  for(const m of models)if(!m.failed){const v=E.step(m.net,train,idx,c.batch,c.lr);if(!Number.isFinite(v.loss))m.failed=true;}
 }
 console.log(name,'seed',seed,models.map(m=>{
  if(m.failed)return m.arm+': diverged';
  const q=E.evaluate(m.net,test,false),b=m.arm==='bn'?E.evaluate(m.net,test,true,c.batch):null;
  return m.arm+': '+(100*(1-q.err)).toFixed(1)+'% / loss '+q.loss.toFixed(3)+(b?' / test-batch '+(100*(1-b.err)).toFixed(1)+'%':'');
 }).join(' | '));
}

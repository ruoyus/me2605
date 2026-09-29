/* Training stays off the UI thread. No external runtime or network calls. */
importScripts('engine.js');
self.onmessage = function(event){
  const c=event.data, E=NormEngine;
  const train=E.moons(c.seed,768,c.scale), test=E.moons(c.seed+100,256,c.scale);
  const models=['none','ln','bn'].map(arm=>({arm,net:E.create(c,arm),history:[],failed:false,lastTrain:null}));
  const grid={X:new Float64Array(40*28*2),y:new Float64Array(40*28),n:40*28};
  for(let j=0;j<28;j++) for(let i=0;i<40;i++){
    let k=j*40+i;grid.X[2*k]=-1.5+4*i/39;grid.X[2*k+1]=(1.7-3*j/27)*c.scale;
  }
  function report(step){
    const rows=models.map(m=>{
      if(!m.failed){
        const v=E.evaluate(m.net,test,false);
        if(!Number.isFinite(v.loss) || !v.scale.every(Number.isFinite)){m.failed=true;m.failedAt=step;}
        else{
          m.latest={loss:v.loss,accuracy:1-v.err,scale:v.scale};
          m.history.push({step,loss:v.loss,accuracy:1-v.err,trainLoss:m.lastTrain});
        }
      }
      let diagnostic=null, map=null;
      if(!m.failed && (step===0 || step===c.steps)){
        map=Array.from(E.evaluate(m.net,grid,false).logits,v=>1/(1+Math.exp(-v)));
      }
      if(!m.failed && m.arm==='bn'){
        const q=E.evaluate(m.net,test,true,c.batch);
        diagnostic={loss:q.loss,accuracy:1-q.err};
      }
      return {arm:m.arm,history:m.history,latest:m.latest,failed:m.failed,failedAt:m.failedAt,
        diagnostic,map,trainLoss:m.lastTrain};
    });
    self.postMessage({step,done:step===c.steps,models:rows,points:step===0 ? Array.from({length:96},(_,i)=>[test.X[2*i],test.X[2*i+1]/c.scale,test.y[i]]) : null});
  }
  try{
    report(0);
    for(let step=1;step<=c.steps;step++){
      const idx=E.indices(c.seed,step,c.batch,train.n);
      for(const m of models){
        if(m.failed) continue;
        const q=E.step(m.net,train,idx,c.batch,c.lr);m.lastTrain=q.loss;
        if(!Number.isFinite(q.loss)){m.failed=true;m.failedAt=step;}
      }
      if(step%20===0 || step===c.steps) report(step);
    }
  }catch(error){self.postMessage({error:error.message});}
};

/* Run with: node test-engine.cjs. No dependencies. */
const assert=require('node:assert/strict');
const E=require('./engine.js');
const cfg={depth:2,width:4,seed:7,placement:'post',affine:true};
const data=E.moons(91,8,1),idx=Int32Array.from([0,1,2,3,4,5,6,7]);
const close=(a,b,tol=2e-5)=>assert.ok(Math.abs(a-b)<tol*(1+Math.abs(a)+Math.abs(b)),`${a} ≠ ${b}`);
// Finite differences: every hidden and output W, b, γ, β in both placements.
let checks=0,worst=0;
for(const placement of ['pre','post']) for(const arm of ['none','ln','bn']){
  const net=E.create({...cfg,placement},arm);
  net.layers.forEach(ly=>{if(ly.gam)ly.gam.forEach((_,j)=>{ly.gam[j]=.8+.13*j;ly.bet[j]=.04*j;});});
  E.forward(net,data,idx,8,false,{batchStats:true});
  const derivatives=net.layers.map(ly=>Object.fromEntries(['W','b','gam','bet'].filter(k=>ly[k]).map(k=>[k,Array.from(ly['g'+k])])));
  for(let l=0;l<net.layers.length;l++)for(const key of Object.keys(derivatives[l])){
    const array=net.layers[l][key];
    for(let i=0;i<array.length;i++){
      const v=array[i],h=1e-6;
      array[i]=v+h;const plus=E.forward(net,data,idx,8,false,{scaleOnly:true,batchStats:true}).loss;
      array[i]=v-h;const minus=E.forward(net,data,idx,8,false,{scaleOnly:true,batchStats:true}).loss;
      array[i]=v;const numeric=(plus-minus)/(2*h),analytic=derivatives[l][key][i];
      worst=Math.max(worst,Math.abs(numeric-analytic));close(numeric,analytic);checks++;
    }
  }
}
// Matched initial weights even though normalization has extra parameters.
const nets=['none','ln','bn'].map(arm=>E.create(cfg,arm));
for(let l=0;l<nets[0].layers.length;l++)for(const net of nets.slice(1))assert.deepEqual(net.layers[l].W,nets[0].layers[l].W);
// Evaluation must neither change weights nor running estimates; singleton inference works.
const bn=nets[2];E.step(bn,data,idx,8,.01);
const before=JSON.stringify(bn);E.evaluate(bn,data,false);E.evaluate(bn,data,true,2);assert.equal(JSON.stringify(bn),before);
// ggam/gbet are reset during evaluation in the old core: evaluation must be fully read-only.
const single={X:data.X.slice(0,2),y:data.y.slice(0,1),n:1};
close(E.evaluate(bn,single,false).logits[0],E.evaluate(bn,data,false).logits[0]);
close(E.evaluate(nets[1],single,false).logits[0],E.evaluate(nets[1],data,false).logits[0]);
assert.throws(()=>E.step(bn,single,Int32Array.of(0),1,.01),/at least two/);
// Population variance for training, unbiased variance for running estimates.
const ly={gam:Float64Array.of(1),bet:Float64Array.of(0)};
const n=E.normForward('bn',Float64Array.of(1,3),2,1,ly,true);
close(n.out[0],-1/Math.sqrt(1+E.eps));close(ly.runningMean[0],.2);close(ly.runningVar[0],1.1);
// Mathematical B=1 output is beta; the demo is not a training implementation.
const singleton=E.normForward('bn',Float64Array.of(3),1,1,{gam:Float64Array.of(2),bet:Float64Array.of(5)},false,true);close(singleton.out[0],5);
// Dataset transformation preserves labels and latent points exactly.
const plain=E.moons(7,12,1),scaled=E.moons(7,12,100);assert.deepEqual(plain.y,scaled.y);
for(let i=0;i<12;i++){close(plain.X[2*i],scaled.X[2*i]);close(plain.X[2*i+1]*100,scaled.X[2*i+1]);}
console.log(`PASS: ${checks} finite differences; max absolute error ${worst.toExponential(2)}; shared initialization, BN running statistics, no evaluation mutation, singleton inference, dataset scaling.`);

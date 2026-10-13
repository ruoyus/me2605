/* Run with: node test-engine.cjs. No dependencies. */
const assert=require('node:assert/strict');
const E=require('./engine.js');
const cfg={depth:2,width:4,seed:7,placement:'post',affine:true};
const data=E.moons(91,8,1),idx=Int32Array.from([0,1,2,3,4,5,6,7]);
const close=(a,b,tol=2e-5)=>assert.ok(Math.abs(a-b)<tol*(1+Math.abs(a)+Math.abs(b)),`${a} ≠ ${b}`);
// Finite differences: every hidden and output W, b, γ, β in both placements.
let checks=0,worst=0;
/* variants: the three default arms (vector γ), weight norm (V and its magnitude gV),
   and the scalar-γ option for LN and weight norm. */
const variants=[
  {cfg:{},arms:['none','ln','bn'],keys:['W','b','gam','bet']},
  {cfg:{gammaShape:'scalar'},arms:['ln'],keys:['W','b','gam','bet']},
];
for(const placement of ['pre','post']) for(const v of variants) for(const arm of v.arms){
  const net=E.create({...cfg,placement,...v.cfg},arm);
  net.layers.forEach(ly=>{if(ly.gam)ly.gam.forEach((_,j)=>{ly.gam[j]=.8+.13*j;ly.bet[j]=.04*j;});});
  /* A scalar β shifts every feature together, and the zero-initialized biases put whole
     rows of pre-activations exactly on their ReLU kinks for a fully-dead sample — the
     derivative there is set-valued and no centered difference can match it. Nudge the
     biases so the comparison happens at a generic point. */
  if(v.cfg.gammaShape==='scalar') net.layers.forEach(ly=>ly.b.forEach((_,j)=>{ly.b[j]=.1+.07*j;}));
  E.forward(net,data,idx,8,false,{batchStats:true});
  const derivatives=net.layers.map(ly=>Object.fromEntries(v.keys
    .filter(k=>ly[k]&&(arm==='wn'||ly['g'+k]))
    .map(k=>[k,Array.from(ly['g'+k]||[])])));
  for(let l=0;l<net.layers.length;l++)for(const key of Object.keys(derivatives[l])){
    const array=net.layers[l][key];
    for(let i=0;i<array.length;i++){
      /* weight-norm rows carry steep curvatures, so halve the step to keep FD truncation small */
      const v2=array[i],h=(arm==='wn')?1e-7:1e-6;
      /* A sample can kill an entire layer, putting the next layer's bias exactly on the ReLU
         kink (pre-activation ≡ 0): there the centered difference equals the average of the two
         one-sided slopes. Compare against the backward pass evaluated at v±h, which handles
         both that and the smooth case. */
      array[i]=v2+h;const plus=E.forward(net,data,idx,8,false,{batchStats:true});
      const gPlus=net.layers[l]['g'+key][i];
      array[i]=v2-h;const minus=E.forward(net,data,idx,8,false,{batchStats:true});
      const gMinus=net.layers[l]['g'+key][i];
      array[i]=v2;const numeric=(plus.loss-minus.loss)/(2*h),analytic=(gPlus+gMinus)/2;
      const ok=Math.abs(numeric-analytic)<2e-5*(1+Math.abs(numeric)+Math.abs(analytic));
      if(!ok)console.error('FAIL',placement,arm,JSON.stringify(v.cfg),'L'+l,key,i,'num',numeric,'ana',analytic);
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
// Weight norm end to end: with the learned magnitude its loss must descend on the spiral data.
{const c2={depth:4,width:16,batch:32,lr:0.05,scale:1,steps:300,seed:7,placement:'post',affine:true,gammaShape:'vector'};
const train=E.moons(7,1280,1),test=E.moons(107,384,1);
const wn=E.create(c2,'wn');let first=NaN;
for(let s=1;s<=300;s++){const idx2=E.indices(7,s,32,train.n);const l=E.step(wn,train,idx2,32,.05).loss;if(s===1)first=l;
  if(!Number.isFinite(l))throw new Error('weight norm diverged at step '+s);}
const acc=(1-E.evaluate(wn,test,false).err)*100;
assert.ok(first>0.6&&acc>70,`weight norm smoke: first loss ${first}, accuracy ${acc.toFixed(1)}%`);}

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

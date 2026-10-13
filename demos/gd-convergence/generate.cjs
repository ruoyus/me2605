/* GD convergence across data distributions.
   Datasets are drawn from several probability distributions; the model is linear least
   squares on [1, x, y] with the ±1 class label as the target, trained by full-batch
   gradient descent at eta = 1/L, L = lambda_max(Phi^T Phi / n). The loss is exactly
   quadratic, so the measured convergence rate IS (1 - lambda_i/L) per step and the
   slowest direction is governed by kappa = L/lambda_min.
   Each run records the curve until ||grad L||_2 <= 1e-3 or 1e6 steps, whichever first.
   Output: curves.json — template.html + this JSON build index.html. */

'use strict';
const fs = require('fs');

/* ---------- RNG (same mulberry32 as the other labs) ---------- */
function mulberry32(seed){var a=seed>>>0;return function(){a|=0;a=a+0x6D2B79F5|0;
  var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function makeGauss(r){return function(){var u=0,v=0;while(u===0)u=r();while(v===0)v=r();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};}

/* ---------- datasets ---------- */
const N=400, HALF=N/2, SEED=7;
function alloc(){return {X:new Float64Array(N*2),t:new Float64Array(N),k:0};}
function push(d,x,y,lab){if(d.k>=N)return;d.X[2*d.k]=x;d.X[2*d.k+1]=y;d.t[d.k]=lab;d.k++;}
function shuffle(d,r){for(var a=N-1;a>0;a--){var j=Math.floor(r()*(a+1)),t=d.t[a];d.t[a]=d.t[j];d.t[j]=t;
  for(var f=0;f<2;f++){t=d.X[2*a+f];d.X[2*a+f]=d.X[2*j+f];d.X[2*j+f]=t;}}return d;}

function blobs(scaleY){return function(r,g){var d=alloc();
  for(var i=0;i<HALF;i++){
    push(d,-1.2+0.5*g(),scaleY*0.5*g(),-1);
    push(d, 1.2+0.5*g(),scaleY*0.5*g(), 1);}
  return shuffle(d,r);};}

const DATASETS=[
 {key:'blobs',name:'Two isotropic Gaussians',note:'means (±1.2, 0), σ = 0.5 · label ±1',color:'#1d4ed8',gen:blobs(1)},
 {key:'uniform',name:'Two uniform squares',note:'side 2, centres (±0.5, 0) · label ±1',color:'#007b72',
  gen(r,g){var d=alloc();for(var i=0;i<HALF;i++){
    push(d,-0.5+2*(r()-0.5),2*(r()-0.5),-1);push(d,0.5+2*(r()-0.5),2*(r()-0.5),1);}return shuffle(d,r);}},
 {key:'laplace',name:'Two Laplace clouds',note:'Laplace(0,1) coordinates, x shifted ±0.8 · label ±1',color:'#b45309',
  gen(r,g){function lap(){var u=r()-0.5;return u<0?Math.log(1+2*u):-Math.log(1-2*u);}
    var d=alloc();for(var i=0;i<HALF;i++){push(d,-0.8+lap(),lap(),-1);push(d,0.8+lap(),lap(),1);}return shuffle(d,r);}},
 {key:'ring-blob',name:'Gaussian blob vs offset ring',note:'blob N(0,0.35²) vs ring r=1.5±0.15 centred (0.8, 0.4) · label ±1',color:'#7c3aed',
  gen(r,g){var d=alloc();for(var i=0;i<HALF;i++){
    push(d,0.35*g(),0.35*g(),-1);
    var t=r()*2*Math.PI,rad=1.5+0.15*g();push(d,0.8+rad*Math.cos(t),0.4+rad*Math.sin(t),1);}return shuffle(d,r);}},
 {key:'moons',name:'Two moons (classic)',note:'the standard interlocking moons, noise 0.1 · label ±1',color:'#0e7490',
  gen(r,g){var d=alloc();for(var i=0;i<HALF;i++){var t=r()*Math.PI;
    push(d,Math.cos(t)+0.1*g(),Math.sin(t)+0.1*g(),-1);
    push(d,1-Math.cos(t)+0.1*g(),0.5-Math.sin(t)+0.1*g(),1);}return shuffle(d,r);}},
 {key:'scale-1e1',name:'Gaussians, second coordinate × 10',note:'same blobs, y stretched tenfold',color:'#d97706',gen:blobs(10)},
 {key:'scale-1e2',name:'Gaussians, second coordinate × 100',note:'same blobs, y stretched a hundredfold',color:'#dc2626',gen:blobs(100)},
 {key:'scale-1e3',name:'Gaussians, second coordinate × 1000',note:'same blobs, y stretched a thousandfold',color:'#6b21a8',gen:blobs(1000)},
 {key:'scale-3e3',name:'Gaussians, second coordinate × 3000',note:'same blobs, y stretched three-thousandfold',color:'#9f1239',gen:blobs(3000)},
];

/* ---------- model: least squares on [1, x, y] ---------- */
const DIM=3;
function features(X,n){var F=new Float64Array(n*DIM);
  for(var i=0;i<n;i++){F[i*DIM]=1;F[i*DIM+1]=X[2*i];F[i*DIM+2]=X[2*i+1];}
  return F;}
function lossGrad(F,t,w,n){
  var loss=0,gr=new Float64Array(DIM);
  for(var i=0;i<n;i++){
    var s=0;for(var j=0;j<DIM;j++)s+=F[i*DIM+j]*w[j];
    var res=s-t[i];loss+=res*res;
    for(j=0;j<DIM;j++)gr[j]+=res*F[i*DIM+j];
  }
  for(j=0;j<DIM;j++)gr[j]/=n;
  return {loss:loss/(2*n),grad:gr};}
function mv(A,v){var w=new Array(DIM).fill(0);
  for(var i=0;i<DIM;i++)for(var j=0;j<DIM;j++)w[i]+=A[i][j]*v[j];return w;}
function norm(v){var s=0;for(const x of v)s+=x*x;return Math.sqrt(s);}
function powerLmax(G){var v=Array.from({length:DIM},(_,i)=>1/(1+i));
  for(var it=0;it<300000;it++){var w=mv(G,v),nw=norm(w);
    if(nw===0)return 0;for(var i=0;i<DIM;i++)w[i]/=nw;
    var diff=0;for(i=0;i<DIM;i++)diff+=Math.abs(w[i]-v[i]);
    v=w;if(diff<1e-15&&it>50)break;}
  return norm(mv(G,v));}
function powerLmin(G,lmax){var c=lmax*1.0000001,M=G.map((row,i)=>row.map((x,j)=>(i===j?c:0)-x));
  var v=Array.from({length:DIM},(_,i)=>1/(1+i));
  for(var it=0;it<300000;it++){var w=mv(M,v),nw=norm(w);
    if(nw===0)return 0;for(var i=0;i<DIM;i++)w[i]/=nw;
    var diff=0;for(i=0;i<DIM;i++)diff+=Math.abs(w[i]-v[i]);
    v=w;if(diff<1e-14&&it>50)break;}
  return Math.max(c-norm(mv(M,v)),1e-300);}

/* ---------- run ---------- */
const TARGET=1e-3, MAXSTEPS=1e6;
const out=[];
for(const ds of DATASETS){
  const r=mulberry32(SEED), g=makeGauss(r);
  const data=ds.gen(r,g);
  const F=features(data.X,N), t=Float64Array.from(data.t);
  const Gm=[];for(let a=0;a<DIM;a++)Gm.push(new Array(DIM).fill(0));
  for(let i=0;i<N;i++)for(let a=0;a<DIM;a++){const fa=F[i*DIM+a];
    for(let b=0;b<DIM;b++)Gm[a][b]+=fa*F[i*DIM+b];}
  for(let a=0;a<DIM;a++)for(let b=0;b<DIM;b++)Gm[a][b]/=N;
  const lmax=powerLmax(Gm), lmin=powerLmin(Gm,lmax);
  const eta=1/lmax, kappa=lmax/lmin;
  const w=new Float64Array(DIM);
  var curve, next=2, reached=false, lastLoss=NaN, lastG=NaN;
  {const lg=lossGrad(F,t,w,N);lastLoss=lg.loss;var gn=0;
   for(let j=0;j<DIM;j++)gn+=lg.grad[j]*lg.grad[j];lastG=Math.sqrt(gn);
   curve=[[0,lastLoss,lastG]];
   if(lastG<=TARGET)reached=true;}
  for(var step=1;step<=MAXSTEPS && !reached;step++){
    const lg=lossGrad(F,t,w,N);lastLoss=lg.loss;
    var gn2=0;for(let j=0;j<DIM;j++)gn2+=lg.grad[j]*lg.grad[j];
    lastG=Math.sqrt(gn2);
    /* (step-1, ...) : the gradient measured after step-1 completed updates */
    if(step>=next){curve.push([step-1,lastLoss,lastG]);next=Math.ceil(next*1.03);}
    if(lastG<=TARGET){curve.push([step-1,lastLoss,lastG]);reached=true;break;}
    for(let j=0;j<DIM;j++)w[j]-=eta*lg.grad[j];
  }
  if(!reached){const lg=lossGrad(F,t,w,N);lastLoss=lg.loss;
    var gn3=0;for(let j=0;j<DIM;j++)gn3+=lg.grad[j]*lg.grad[j];lastG=Math.sqrt(gn3);
    curve.push([MAXSTEPS,lastLoss,lastG]);}
  const pts=[];for(let i=0;i<240;i++){const k=Math.floor(i*N/240);
    pts.push([+data.X[2*k].toFixed(4),+data.X[2*k+1].toFixed(4),data.t[k]>0?1:0]);}
  out.push({key:ds.key,name:ds.name,note:ds.note,color:ds.color,n:N,kappa:+kappa.toPrecision(3),
    eta:eta,steps:curve[curve.length-1][0],reached,
    finalLoss:+lastLoss.toPrecision(4),finalGrad:+lastG.toPrecision(3),curve,points:pts});
  console.log(ds.key.padEnd(9),'κ='+kappa.toExponential(2),'η='+eta.toExponential(2),
    reached?('steps='+step):('NOT reached in 1e6'),
    'loss='+lastLoss.toExponential(3),'||g||='+lastG.toExponential(3));
}
fs.writeFileSync(__dirname+'/curves.json',JSON.stringify({target:TARGET,maxSteps:MAXSTEPS,datasets:out}));
console.log('wrote curves.json');

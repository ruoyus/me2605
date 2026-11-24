"use strict";
const byId = id => document.getElementById(id);
const el = Object.fromEntries(["play","reset","end","max-steps","seed","step","progress","fruit","vegetable","query","swap","heldout","split","prefix","target","prediction","prob","mlp-prediction","mlp-prob","mlp-error","error-status","trace","top","loss","metrics","view","vectors","projectionCaption","heat","ideal","extent","scoreDetail","witness","vocab"].map(id => [id, byId(id)]));
let run = window.RUNS[7], frameIndex = 0, timer = null, swapped = false, heldoutIndex = -1;
const lastIndex = () => run.checkpoints.findIndex(frame => frame.step === Number(el["max-steps"].value));
const lastStep = () => run.checkpoints[lastIndex()].step;

function setStepOptions(limit) {
  const steps = run.checkpoints.slice(1).map(frame => frame.step);
  if (!steps.length) steps.push(0);
  el["max-steps"].replaceChildren(...steps.map(step => new Option(step.toLocaleString("en-US"), step)));
  el["max-steps"].value = steps.find(step => step >= limit) ?? steps.at(-1);
}

const mlpCheckpoints = () => window.MLP_RUNS?.[run.seed]?.checkpoints ?? [];
const currentMLP = () => mlpCheckpoints().find(frame => frame.step === run.checkpoints[frameIndex].step);
const words = run.words;
const ids = Object.fromEntries(words.map((word, i) => [word, i]));
const groups = [run.fruits, run.vegetables];
const queryWords = ["fruit", "vegetable"];
const picked = ["fruit", "vegetable", "apple", "pear", "carrot", "cabbage"];
const palette = { blue:"#2756a6", rose:"#a43453", green:"#216551", gray:"#d4dbe1", ink:"#344352" };
const dot = (x, y) => x.reduce((sum, value, i) => sum + value * y[i], 0);
const apply = (matrix, vector) => matrix.map(row => dot(row, vector));
const length = x => Math.sqrt(dot(x,x));
const e = x => x === 0 ? "0" : x.toExponential(2);
const fmt = x => x.toFixed(2).replace("-", "−");
const pct = x => x > 0.999999 ? ">99.9999%" : (100*x).toFixed(x > 0.9999 ? 4 : 2) + "%";
const isHeldout = (i,j) => ((j - 7*i + 150) % 30) < 6;
const heldout = groups[0].flatMap((_,i) => groups[1].flatMap((_,j) => isHeldout(i,j) ? [[i,j]] : []));

for (const [select, labels] of [[el.fruit,groups[0]],[el.vegetable,groups[1]]]) {
  labels.forEach(word => select.add(new Option(word,word)));
}

function canvasContext(canvas) {
  const {width, height} = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width*dpr);
  canvas.height = Math.round(height*dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr,dpr);
  ctx.font = '12px system-ui';
  return {ctx,w:width,h:height};
}

function plotLoss() {
  const {ctx,w,h} = canvasContext(el.loss);
  const last = lastStep();
  const step = run.checkpoints[frameIndex].step;
  const margin = {left:48,right:17,top:12,bottom:38};
  const x = update => margin.left + update/Math.max(last,1)*(w-margin.left-margin.right);
  const mlp = mlpCheckpoints().filter(frame => frame.step <= last);
  const frames = [...run.checkpoints.slice(0,lastIndex()+1), ...mlp];
  const floor = Math.min(0,Math.floor(Math.min(...frames.map(f=>Math.log10(Math.min(f.train.ce,f.test.ce))))));
  const ceiling = Math.ceil(Math.max(...frames.map(f=>Math.log10(Math.max(f.train.ce,f.test.ce)))));
  const y = log => margin.top + (ceiling-log)/(ceiling-floor)*(h-margin.top-margin.bottom);
  ctx.fillStyle = "#546474";
  for (let log = ceiling; log >= floor; log--) {
    const yy = y(log);
    ctx.strokeStyle = "#e8ecec";
    ctx.beginPath();ctx.moveTo(x(0),yy);ctx.lineTo(x(last),yy);ctx.stroke();
    if (log % 2 === 0 || log === floor || log === ceiling) ctx.fillText("1e"+log,8,yy+4);
  }
  for (const update of new Set([0, .25, .5, .75, 1].map(part => Math.round(part*last)))) {
    ctx.textAlign = update===last ? "right" : "center";
    ctx.fillText(String(update),x(update),h-20);
  }
  ctx.textAlign="center";
  ctx.fillText("Adam update",w/2,h-5);
  for (const [dashed, checkpoints] of [[true,mlp],[false,run.checkpoints]]) {
    const visible = checkpoints.filter(frame => frame.step <= step);
    if (!visible.length) continue;
    for (const [key, color] of [["train",palette.blue],["test",palette.rose]]) {
      ctx.setLineDash(dashed?[6,5]:[]);
      ctx.strokeStyle=color;ctx.lineWidth=2.4;ctx.beginPath();
      for (const [i, f] of visible.entries()) {
        const xx=x(f.step), yy=y(Math.log10(f[key].ce));
        i ? ctx.lineTo(xx,yy) : ctx.moveTo(xx,yy);
      }
      ctx.stroke();
      const f=visible.at(-1);
      const xx=x(f.step), yy=y(Math.log10(f[key].ce));
      ctx.fillStyle=color;
      if(dashed) ctx.fillRect(xx-3,yy-3,6,6);
      else {ctx.beginPath();ctx.arc(xx,yy,4.2,0,Math.PI*2);ctx.fill();}
    }
  }
}

function plotVectors() {
  const {ctx,w,h} = canvasContext(el.vectors);
  const view = el.view.value;
  const frame = run.checkpoints[frameIndex][view];
  const points = run.checkpoints.flatMap(f=>f[view]);
  const low = [Math.min(...points.map(p=>p[0])),Math.min(...points.map(p=>p[1]))];
  const high = [Math.max(...points.map(p=>p[0])),Math.max(...points.map(p=>p[1]))];
  const scale = Math.min((w-95)/(high[0]-low[0]),(h-50)/(high[1]-low[1]));
  const x = value => w/2 + (value-(low[0]+high[0])/2)*scale;
  const y = value => h/2 - (value-(low[1]+high[1])/2)*scale;
  ctx.strokeStyle="#e3e8e8";ctx.lineWidth=1;ctx.beginPath();
  ctx.moveTo(x(0),9);ctx.lineTo(x(0),h-22);ctx.moveTo(9,y(0));ctx.lineTo(w-9,y(0));ctx.stroke();
  ctx.fillStyle="#546474";ctx.fillText("PC 2",9,16);ctx.fillText("PC 1",w-36,h-8);
  for (const [a,b] of frame) {
    ctx.fillStyle=palette.gray;ctx.beginPath();ctx.arc(x(a),y(b),2.6,0,2*Math.PI);ctx.fill();
  }
  const labelFrames=[];
  picked.forEach((word,p) => {
    const index=ids[word]-3;
    const color=p<2?palette.blue:p<4?palette.rose:palette.green;
    ctx.globalAlpha=.48; ctx.strokeStyle=color;ctx.lineWidth=1.65;ctx.beginPath();
    for (let i=0;i<=frameIndex;i++) {
      const [a,b]=run.checkpoints[i][view][index];
      i?ctx.lineTo(x(a),y(b)):ctx.moveTo(x(a),y(b));
    }
    ctx.stroke();ctx.globalAlpha=1;
    const [a,b]=frame[index],xx=x(a),yy=y(b);
    ctx.fillStyle=color;ctx.beginPath();
    if(p<2){ctx.moveTo(xx,yy-6);ctx.lineTo(xx+6,yy);ctx.lineTo(xx,yy+6);ctx.lineTo(xx-6,yy);ctx.closePath();}
    else ctx.arc(xx,yy,4,0,2*Math.PI);
    ctx.fill();
    const textWidth=ctx.measureText(word).width;
    const labelX=Math.min(Math.max(xx+8,8),w-textWidth-9);
    let labelY=Math.min(h-17,Math.max(14,yy-5));
    for(let retry=0;retry<6 && labelFrames.some(([qx,qy,qw])=>Math.abs(qy-labelY)<14 && labelX<qx+qw+3 && labelX+textWidth>qx-3);retry++) labelY=labelY+15>h-14 ? Math.max(14,labelY-27) : labelY+15;
    labelFrames.push([labelX,labelY,textWidth]);
    ctx.fillStyle="rgba(255,255,255,.93)";ctx.fillRect(labelX-2,labelY-11,textWidth+4,14);
    ctx.fillStyle=color;ctx.fillText(word,labelX,labelY);
  });
  el.projectionCaption.textContent = "Fixed PCA plane and axes across this run. " + (view==="embedding" ? "The two query words and four items are shown before projection through WQ or WK." : "Query words go through WQ. Produce words go through WK. Scores use the full 16 dimensions.");
}

function probability(logits) {
  const max=Math.max(...logits);
  const exp=logits.map(l=>Math.exp(l-max));
  const sum=exp.reduce((a,b)=>a+b,0);
  return exp.map(v=>v/sum);
}

function mlpLogits(prefix, parameters) {
  const [E,W1,b1,W2,b2]=parameters;
  const input=prefix.flatMap(word=>E[ids[word]]);
  const hidden=apply(W1,input).map((v,i)=>Math.max(0,v+b1[i]));
  return apply(W2,hidden).map((v,i)=>v+b2[i]);
}

function best(logits) {
  return logits.reduce((winner,value,i)=>value>logits[winner]?i:winner,0);
}

function probe(parameters, mlpParameters) {
  const [E,WQ,WK,WV,H] = parameters;
  const fruit=el.fruit.value, vegetable=el.vegetable.value, query=el.query.value;
  const items=swapped ? [vegetable,fruit] : [fruit,vegetable];
  const prefix=["items",...items,"name","the",query];
  const q=apply(WQ,E[ids[query]]);
  const context=Array(q.length).fill(0);
  const trace=prefix.slice(0,5).map((word,i) => {
    const v=apply(WV,E[ids[word]]);
    const score=dot(q,apply(WK,E[ids[word]]));
    v.forEach((value,k)=>context[k]+=score*value);
    return [i+1,word,fmt(score),fmt(Math.abs(score)*length(v))];
  });
  const p=probability(apply(H,context));
  const rank=p.map((_,i)=>i).sort((a,b)=>p[b]-p[a]);
  const target=query==="fruit"?fruit:vegetable;
  el.split.textContent=isHeldout(groups[0].indexOf(fruit),groups[1].indexOf(vegetable))?"HELD-OUT PAIR":"TRAINING PAIR";
  const tokens=["Items:",items[0]+",",items[1]+".","Name","the",query+":"];
  el.prefix.replaceChildren(...tokens.map((word,i)=>{
    const span=document.createElement("span");
    span.textContent=word;
    if(i===1||i===2)span.className="item";
    if(i===5)span.className="query";
    return span;
  }));
  el.target.textContent=target;
  el.prediction.textContent=words[rank[0]];
  el.prediction.style.color=words[rank[0]]===target?palette.green:palette.rose;
  el.prob.textContent=pct(p[ids[target]]);
  if (mlpParameters) {
    const mlp=mlpLogits(prefix,mlpParameters);
    const winner=best(mlp);
    el["mlp-prediction"].textContent=words[winner];
    el["mlp-prediction"].style.color=winner===ids[target]?palette.green:palette.rose;
    el["mlp-prob"].textContent=pct(probability(mlp)[ids[target]]);
  } else {
    el["mlp-prediction"].textContent="No checkpoint";
    el["mlp-prediction"].style.color="";
    el["mlp-prob"].textContent="—";
  }
  el.trace.innerHTML=trace.map(row=>"<tr>"+row.map(v=>`<td>${v}</td>`).join("")+"</tr>").join("");
  el.top.textContent="Top predictions · "+rank.slice(0,4).map(i=>words[i]+" "+pct(p[i])).join(" · ");
}

const blocks=[];
for (let start=0;start<50;start+=25) {
  const grid=document.createElement("div");grid.className="heatgrid";
  const blank=document.createElement("span");
  const group=document.createElement("span");group.className="group";group.textContent=start===0?"Items 1–25 · apple to cauliflower":"Items 26–50 · spinach to chard";
  grid.append(blank,group);
  const cells=[];
  queryWords.forEach((query,q)=>{
    const label=document.createElement("span");label.className="queryname";label.textContent=query;grid.append(label);
    for(let index=start;index<start+25;index++) {
      const word=words[index+5];
      const button=document.createElement("button");
      button.addEventListener("click",()=>{
        el.query.value=query;
        (index<20?el.fruit:el.vegetable).value=word;
        render();
        byId("probe-title").scrollIntoView({block:"start"});
      });
      grid.append(button);cells.push({button,index,q,word,query});
    }
  });
  grid.append(document.createElement("span"));
  for(let index=start;index<start+25;index++){
    const small=document.createElement("small");small.textContent=words[index+5].slice(0,3);small.title=words[index+5];grid.append(small);
  }
  el.heat.append(grid);blocks.push(...cells);
}
function mix(hex,t){
  const v=hex.match(/\w\w/g).map(n=>parseInt(n,16));
  return "rgb("+v.map(c=>Math.round(247+(c-247)*t)).join(",")+")";
}
function scores(parameters){
  const [E,WQ,WK]=parameters;
  const queries=E.slice(3,5).map(v=>apply(WQ,v));
  const keys=E.slice(5).map(v=>apply(WK,v));
  const live=queries.map(q=>keys.map(k=>dot(q,k)));
  const bound = el.ideal.checked ? 1 : Math.max(1,...run.checkpoints.flatMap(f=>{
    if(f._bound===undefined){
      const [e,wq,wk]=f.parameters;
      const kk=e.slice(5).map(v=>apply(wk,v));
      f._bound=Math.max(...e.slice(3,5).flatMap(v=>{
        const qq=apply(wq,v);
        return kk.map(k=>Math.abs(dot(qq,k)));
      }));
    }
    return f._bound;
  }));
  for(const {button,index,q,word,query} of blocks){
    const s=el.ideal.checked ? Number((index<20)===(q===0)) : live[q][index];
    const t=Math.min(1,Math.abs(s)/bound);
    button.style.background=mix(s<0?"45468f":"a43453",t);
    button.style.color=t>.65?"white":"#192632";
    button.textContent=fmt(s);
    button.title=query+" → "+word+" · "+s.toFixed(5);
    button.setAttribute("aria-label",button.title);
  }
  el.extent.textContent="Color range ±"+bound.toFixed(2)+" · fixed across checkpoints";
  const groupMean=(q,a,b)=>live[q].slice(a,b).reduce((s,v)=>s+v,0)/(b-a);
  el.scoreDetail.textContent=`Learned group means · fruit → fruits ${fmt(groupMean(0,0,20))} / vegetables ${fmt(groupMean(0,20,50))} · vegetable → fruits ${fmt(groupMean(1,0,20))} / vegetables ${fmt(groupMean(1,20,50))}`;
}
function render(){
  const frame=run.checkpoints[frameIndex];
  const mlpFrame=currentMLP();
  el["error-status"].textContent="";
  el["mlp-error"].disabled=!mlpFrame;
  el.play.disabled=lastIndex()<1;
  el.step.textContent=frame.step.toLocaleString("en-US")+" / "+lastStep().toLocaleString("en-US");
  el.progress.max=lastIndex();
  el.progress.value=frameIndex;
  el.progress.setAttribute("aria-valuetext","Update "+frame.step);
  el.metrics.innerHTML=[["Attention",frame],["MLP",mlpFrame]].filter(([, checkpoint])=>checkpoint).flatMap(([model,checkpoint])=>["train","test"].map(k=>{
    const m=checkpoint[k];
    return `<tr><th scope="row">${model}</th><td>${k==="train"?"Train":"Test"}</td><td>${e(m.ce)}</td><td>${e(m.pplMinusOne)}</td><td>${(100*m.accuracy).toFixed(1)}%</td></tr>`;
  })).join("");
  probe(frame.parameters,mlpFrame?.parameters);scores(frame.parameters);plotLoss();plotVectors();
}
function pause(){
  clearInterval(timer);timer=null;el.play.textContent="▶ Play training";
}
el.play.addEventListener("click",()=>{
  if(timer){pause();return;}
  if(frameIndex===lastIndex()){frameIndex=0;render();}
  el.play.textContent="❚❚ Pause";
  timer=setInterval(()=>{
    frameIndex++;render();
    if(frameIndex===lastIndex())pause();
  },200);
});
el.reset.onclick=()=>{pause();frameIndex=0;render();};
el.end.onclick=()=>{pause();frameIndex=lastIndex();render();};
el.progress.oninput=()=>{pause();frameIndex=Number(el.progress.value);render();};
el.seed.onchange=()=>{
  pause();
  const limit=lastStep();
  run=window.RUNS[el.seed.value];
  setStepOptions(limit);
  frameIndex=0;
  render();
};
el["max-steps"].onchange=()=>{
  pause();
  frameIndex=Math.min(frameIndex,lastIndex());
  render();
};
el.swap.onclick=()=>{swapped=!swapped;render();};
el.heldout.onclick=()=>{heldoutIndex=(heldoutIndex+1)%heldout.length;const [i,j]=heldout[heldoutIndex];el.fruit.value=groups[0][i];el.vegetable.value=groups[1][j];render();};
el["mlp-error"].onclick=()=>{
  pause();
  const parameters=currentMLP()?.parameters;
  if (!parameters) return;
  for(const [i,j] of heldout){
    const fruit=groups[0][i], vegetable=groups[1][j];
    for(const [first,second] of [[fruit,vegetable],[vegetable,fruit]]){
      for(const query of queryWords){
        const target=query==="fruit"?fruit:vegetable;
        const logits=mlpLogits(["items",first,second,"name","the",query],parameters);
        if(best(logits)===ids[target]) continue;
        el.fruit.value=fruit;el.vegetable.value=vegetable;el.query.value=query;swapped=first!==fruit;
        render();
        el["error-status"].textContent="Found a held-out prefix the MLP gets wrong at this checkpoint.";
        return;
      }
    }
  }
  el["error-status"].textContent="The MLP gets all 480 held-out prefixes right at this checkpoint.";
};
for(const input of [el.fruit,el.vegetable,el.query,el.ideal,el.view])input.onchange=render;
window.addEventListener("resize",()=>{plotLoss();plotVectors();});
setStepOptions(60);
el.witness.innerHTML=run.witness.map(d=>`<tr><th scope="row">${d.scale.toLocaleString("en-US")}</th><td>${e(d.train.ce)}</td><td>${e(d.test.ce)}</td><td>${(100*d.test.accuracy).toFixed(1)}%</td></tr>`).join("");
el.vocab.textContent="Template and query words: "+words.slice(0,5).join(", ")+". Fruits: "+groups[0].join(", ")+". Vegetables: "+groups[1].join(", ")+".";
render();

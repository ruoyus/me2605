'use strict';
const $=id=>document.getElementById(id), E=NormEngine;
let language='en',worker=null,active=null,result=null,points=[],maps={},running=false;
const colors={none:'#d8543d',ln:'#3262ed',bn:'#008577'}, names={none:'No normalization',ln:'LayerNorm',bn:'BatchNorm'};
const t=(en,zh)=>language==='zh'?zh:en;
const presets={control:{depth:4,width:16,batch:32,lr:0.03,scale:1,steps:300},scale:{depth:8,width:16,batch:32,lr:0.03,scale:100,steps:300},batch:{depth:4,width:16,batch:2,lr:0.03,scale:1,steps:300}};
let preset='control';
function config(){return {depth:+$('depth').value,width:+$('width').value,batch:+$('batch').value,lr:+$('lr').value,scale:+$('scale').value,steps:+$('steps').value,seed:+$('seed').value,placement:$('placement').value,affine:$('affine').value==='yes'};}
function explain(){
  const c=config();
  $('architecture').textContent=c.placement==='post'?'x → [W, b → ReLU → Norm] × '+c.depth+' → linear → loss':'x → [W, b → Norm → ReLU] × '+c.depth+' → linear → loss';
  const questions={control:t('Predict: on a modest network with well-scaled inputs, do we need normalization to learn? Try learning rates 0.01 and 0.1 after the first run.','预测：网络不深、输入尺度适中时，不用归一化能学好吗？首次运行后再试学习率 0.01 和 0.1。'),scale:t('Predict: what changes when one input coordinate is 100× larger? Run this setting, then change ONLY input scale to 1. Next change ONLY depth.','预测：一个输入坐标放大 100 倍后会怎样？先运行，再只将输入尺度改为 1，然后再只改变层数。'),batch:t('Predict: with only 2 samples per batch, how noisy are BN statistics? Compare BN’s two evaluation rules below. Then change ONLY batch size to 32.','预测：每批只有 2 个样本，BN 的统计量有多不稳定？比较下方的两种 BN 推理规则，再只将批大小改为 32。')};
  $('hypothesis').textContent=questions[preset];
  if(c.width===2) $('hypothesis').textContent+=' '+t('Width 2 is a stress test: LN can remove most within-sample information (exactly so when ε = 0).','宽度 2 是极端测试：LN 可能移除大部分样本内信息（ε = 0 时尤其明显）。');
}
function sandbox(){
  const selected=$('companions').value;
  const samples=selected==='single'?[[1,2,4]]:selected==='a'?[[1,2,4],[3,2,6],[6,5,1]]:[[1,2,4],[10,-2,1],[12,3,9]];
  const B=samples.length,h=Float64Array.from(samples.flat()),ly={gam:new Float64Array([1,1,1]),bet:new Float64Array(3)};
  const ln=E.normForward('ln',h,B,3,ly,false),bn=E.normForward('bn',h,B,3,ly,false,true);
  const f=v=>Number(v).toFixed(3),vec=a=>'['+Array.from(a).slice(0,3).map(f).join(', ')+']';
  $('matrix').innerHTML='<table><thead><tr><th>'+t('Feature','Feature')+'</th>'+samples.map((_,i)=>'<th class="'+(i===0?'fixed':'')+'">'+t('Sample ','样本 ')+(i+1)+'</th>').join('')+'</tr></thead><tbody>'+[0,1,2].map(j=>'<tr><th>'+t('Neuron ','神经元 ')+(j+1)+'</th>'+samples.map((a,i)=>'<td class="'+(i===0?'fixed':'')+'">'+a[j]+'</td>').join('')+'</tr>').join('')+'</tbody></table><p class="small">'+t('Computed: LN has ','计算值：LN 有 ')+B+' μ + '+B+' σ; BN: 3 μ + 3 σ.<br>'+t('Learned: 3 γ + 3 β for either method, regardless of B.','可学习参数：两者均为 3 γ + 3 β，与 B 无关。')+'</p>';
  $('axis-result').innerHTML='<div class="formula ln">LN → '+vec(ln.out)+'</div><div class="formula bn">BN → '+vec(bn.out)+'</div><p class="small">LN: μ₁ = '+f(ln.mu[0])+', σ₁ = '+f(1/ln.inv[0])+'<br>BN: μ = '+vec(bn.mu)+', σ = '+vec(Array.from(bn.inv,v=>1/v))+'</p>'+(B===1?'<div class="warning">'+t('With B = 1, centered values are zero. The mathematical BN output is β, not 1. Standard training implementations reject this fully connected case. Stored-statistics inference can still process one sample.','B = 1 时中心化值为零，数学上的 BN 输出是 β，不是 1。标准实现会拒绝这种全连接层训练情形。使用存储统计量的推理仍可处理单样本。')+'</div>':'<p class="small">'+t('σ includes ε = 10⁻⁵. γ and β are shared, not recomputed for each sample.','σ 包含 ε = 10⁻⁵。γ、β 共享，不为每个样本重新计算。')+'</p>');
}
function canvas(id){const el=$(id),ctx=el.getContext('2d');ctx.clearRect(0,0,el.width,el.height);return {ctx,w:el.width,h:el.height};}
function chart(id,series,xMax,yMin,yMax,log,yLabel,xLabel){
  const {ctx:g,w,h}=canvas(id),left=55,right=17,top=14,bottom=40;
  const tx=x=>left+(w-left-right)*x/Math.max(1,xMax), trans=v=>log?Math.log10(Math.max(1e-8,v)):v;
  const lo=trans(yMin),hi=trans(yMax),ty=y=>h-bottom-(h-top-bottom)*(trans(y)-lo)/(hi-lo||1);
  g.font='13px system-ui';g.lineWidth=1;
  for(let i=0;i<=4;i++){
    const val=log?10**(lo+(hi-lo)*i/4):yMin+(yMax-yMin)*i/4,y=ty(val);
    g.strokeStyle='#e4e9f1';g.beginPath();g.moveTo(left,y);g.lineTo(w-right,y);g.stroke();
    g.fillStyle='#66748a';g.textAlign='right';g.fillText(log?(val>=10?val.toExponential(0):val.toFixed(2)):Math.round(val*100)+'%',left-8,y+4);
    const x=xMax*i/4;g.textAlign='center';g.fillText(Math.round(x),tx(x),h-bottom+19);
  }
  g.fillStyle='#66748a';g.fillText(xLabel,w/2,h-3);
  for(const s of series){g.strokeStyle=colors[s.arm];g.lineWidth=2.7;g.beginPath();let moved=false;for(const p of s.data){if(!Number.isFinite(p[1])){moved=false;continue;}const x=tx(p[0]),y=ty(p[1]);if(!moved){g.moveTo(x,y);moved=true;}else g.lineTo(x,y);}g.stroke();if(s.data.length===1){g.beginPath();g.arc(tx(s.data[0][0]),ty(s.data[0][1]),3,0,7);g.fillStyle=colors[s.arm];g.fill();}}
  if(!series.some(s=>s.data.length)){g.fillStyle='#768399';g.textAlign='center';g.fillText(t('Run a comparison to see measured curves','运行对比，查看实测曲线'),w/2,h/2);}
}
function map(arm){
  const {ctx:g,w,h}=canvas('map-'+arm),data=maps[arm];
  if(!data) return;
  for(let j=0;j<28;j++)for(let i=0;i<40;i++){const p=data[j*40+i];g.fillStyle='rgb('+Math.round(230-90*p)+','+Math.round(235-30*p)+','+Math.round(253-64*p)+')';g.fillRect(i*w/40,j*h/28,w/40+1,h/28+1);}
  for(const p of points){g.beginPath();g.arc((p[0]+1.5)/4*w,(1.7-p[1])/3*h,2.8,0,7);g.fillStyle=p[2]?'#007b72':'#37499d';g.fill();g.strokeStyle='#ffffff';g.lineWidth=.7;g.stroke();}
}
function render(){
  const models=result?result.models:[],steps=active?active.steps:config().steps;
  const losses=models.flatMap(m=>m.history.map(p=>p.loss)).filter(v=>v>0&&Number.isFinite(v));
  const min=losses.length?Math.min(.1,...losses):.1,max=losses.length?Math.max(1,...losses):1;
  chart('loss',models.map(m=>({arm:m.arm,data:m.history.map(p=>[p.step,p.loss])})),steps,min,max,true,'loss',t('SGD updates','SGD 更新次数'));
  chart('accuracy',models.map(m=>({arm:m.arm,data:m.history.map(p=>[p.step,p.accuracy])})),steps,0,1,false,'accuracy',t('SGD updates','SGD 更新次数'));
  const scales=models.filter(m=>m.latest&&!m.failed).flatMap(m=>m.latest.scale).filter(v=>v>0&&Number.isFinite(v));
  chart('signal',models.filter(m=>m.latest&&!m.failed).map(m=>({arm:m.arm,data:m.latest.scale.map((v,i)=>[i+1,v])})),active?active.depth:config().depth,Math.min(.1,...scales),Math.max(2,...scales),true,'RMS',t('Hidden layer','隐藏层'));
  $('cards').innerHTML=['none','ln','bn'].map(arm=>{
    const m=models.find(a=>a.arm===arm),v=m&&m.latest;
    return '<article class="model-card" style="--color:'+colors[arm]+'"><h3>'+names[arm]+'</h3><div class="metric">'+(m&&m.failed?t('Diverged','发散'):v?(100*v.accuracy).toFixed(1)+'%':'—')+'</div><p class="small">'+(m&&m.failed?t('Non-finite values at step ','第 ')+m.failedAt+t('',' 步出现非有限值'):t('Held-out accuracy','留出集准确率'))+'</p><canvas class="map" id="map-'+arm+'" width="280" height="196" role="img" aria-label="'+names[arm]+' decision map"></canvas><p class="small">'+(m&&m.failed?t('Curve stops at the last finite evaluation.','曲线停在最后一个有限评估值。'):v?t('Held-out loss: ','留出集损失：')+v.loss.toFixed(3):t('Decision map appears when training starts.','开始训练后显示分类图。'))+'</p><p class="small">'+(m&&Number.isFinite(m.trainLoss)?t('Last training batch loss: ','最近训练批损失：')+m.trainLoss.toFixed(3):'')+'</p></article>';
  }).join('');
  for(const arm of ['none','ln','bn']) {
    map(arm);
    if(maps[arm]) {
      const caption=document.createElement('p');caption.className='small';
      caption.textContent=result&&result.done?t('Final decision map · dots: held-out labels; background: predicted class probability. Axes show unscaled inputs.','最终分类图 · 点为留出标签，背景为预测类别概率。坐标轴使用缩放前的输入。'):t('Initial decision map (updates after training). Dots: held-out labels.','初始化分类图（训练完成后更新）。点为留出标签。');
      $('map-'+arm).after(caption);
    }
  }
  const bn=models.find(m=>m.arm==='bn');
  if(bn&&bn.latest&&!bn.failed){
    $('inference-result').innerHTML='<table><tr><th>'+t('BN evaluation rule','BN 推理规则')+'</th><th>'+t('Accuracy','准确率')+'</th><th>'+t('Loss','损失')+'</th></tr><tr><td>'+t('Stored training statistics','存储的训练统计量')+'</td><td>'+ (100*bn.latest.accuracy).toFixed(1)+'%</td><td>'+bn.latest.loss.toFixed(3)+'</td></tr><tr><td>'+t('Current test-batch statistics','当前测试批统计量')+'</td><td>'+ (100*bn.diagnostic.accuracy).toFixed(1)+'%</td><td>'+bn.diagnostic.loss.toFixed(3)+'</td></tr></table><p class="small">'+t('At update 0, stored BN means are 0 and variances are 1; the running estimates have not been trained.','更新次数为 0 时，BN 存储均值为 0、方差为 1，尚未学习运行统计量。')+'</p>';
  } else $('inference-result').textContent=bn&&bn.failed?t('BN diverged; no valid final inference comparison.','BN 已发散，无法进行有效的最终推理比较。'):t('Run an experiment to compare the two rules.','先运行实验，再比较两种规则。');
}
function setRunning(value){running=value;$('run').disabled=value;$('stop').disabled=!value;document.querySelectorAll('.controls select,.controls input,[data-preset]').forEach(el=>el.disabled=value);}
function run(){
  const c=config();if(!Number.isInteger(c.seed)||c.seed<1||c.seed>99999){$('status').textContent=t('Seed must be an integer from 1 to 99999.','种子必须是 1–99999 的整数。');return;}
  if(worker)worker.terminate();active=c;result=null;maps={};points=[];setRunning(true);$('export').disabled=true;
  $('progress').max=c.steps;$('progress').value=0;$('status').textContent=t('Training three matched models…','正在训练三个对照模型……');render();
  worker=new Worker('worker.js');
  worker.onerror=event=>{setRunning(false);$('status').textContent=t('Training could not run: ','无法运行训练：')+event.message;worker.terminate();};
  worker.onmessage=event=>{
    if(event.data.error){setRunning(false);$('status').textContent=event.data.error;worker.terminate();return;}
    result=event.data;if(result.points)points=result.points;
    for(const m of result.models){if(m.map)maps[m.arm]=m.map;if(m.failed)delete maps[m.arm];}
    $('progress').value=result.step;
    $('status').textContent=(result.done?t('Finished · ','完成 · '):t('Training · ','训练中 · '))+result.step+' / '+c.steps+t(' updates',' 次更新');
    if(result.done){setRunning(false);$('export').disabled=false;worker.terminate();worker=null;}
    render();
  };worker.postMessage(c);
}
$('run').onclick=run;
$('stop').onclick=()=>{if(worker)worker.terminate();worker=null;setRunning(false);$('export').disabled=!result;$('status').textContent=t('Stopped. Curves show the last completed checkpoint. Run starts fresh.','已停止。曲线保留最后完成的检查点。再次运行将从头开始。');};
$('export').onclick=()=>{if(!result)return;const payload={version:1,config:active,completed:!!result.done,step:result.step,protocol:'768 training / 256 held-out two moons; matched initialization and mini-batches; BN standard evaluation uses running statistics',models:result.models.map(m=>({arm:m.arm,failed:m.failed,failedAt:m.failedAt,history:m.history,latest:m.latest,batchStatisticsDiagnostic:m.diagnostic}))};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='normalization-seed-'+active.seed+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
document.querySelectorAll('[data-preset]').forEach(button=>button.onclick=()=>{preset=button.dataset.preset;Object.entries(presets[preset]).forEach(([key,value])=>$(key).value=value);$('placement').value='post';$('affine').value='yes';document.querySelectorAll('[data-preset]').forEach(b=>b.classList.toggle('selected',b===button));explain();if(!result)render();else $('status').textContent=t('Settings changed. Run to replace the previous results.','设置已改变。点击运行以替换之前的结果。');});
document.querySelectorAll('.controls select,.controls input').forEach(el=>el.addEventListener('change',()=>{explain();if(result)$('status').textContent=t('Settings changed. Displayed results still use the previous run settings.','设置已改变。当前结果仍使用上次运行的设置。');}));
$('companions').onchange=sandbox;
$('lang').onclick=()=>{language=language==='en'?'zh':'en';document.documentElement.lang=language==='en'?'en':'zh-CN';document.querySelectorAll('[data-en]').forEach(el=>el.textContent=el.dataset[language]);$('lang').textContent=language==='en'?'中文':'English';sandbox();explain();render();$('status').textContent=running?t('Training…','训练中……'):result?t('Displayed results: update ','当前结果：更新次数 ')+result.step:t('Ready. Choose a question, then run.','准备就绪。选择问题后运行。');};
sandbox();explain();render();

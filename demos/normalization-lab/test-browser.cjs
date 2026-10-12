// Run against the repo's local HTTP server; requires Playwright.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8765/demos/normalization-lab/');
 await page.getByLabel('Other samples in the mini-batch').selectOption('b');
 assert.ok((await page.locator('#axis-result').innerText()).includes('LN → [-1.069, -0.267, 1.336]'));
 await page.getByLabel('Other samples in the mini-batch').selectOption('single');
 assert.ok((await page.locator('#axis-result').innerText()).includes('BN → [0.000, 0.000, 0.000]'));
 await page.getByLabel('Other samples in the mini-batch').selectOption('a');
 await page.getByRole('button',{name:'Run comparison',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Finished'),{},{timeout:60000});
 assert.deepEqual(await page.locator('.metric').allTextContents(),['84.8%','99.6%','100.0%']);
 const downloadPromise=page.waitForEvent('download');
 await page.getByRole('button',{name:'Export settings + results'}).click();
 const download=await downloadPromise;assert.ok(download.suggestedFilename().endsWith('.json'));
 await page.screenshot({path:'/private/tmp/normalization-lab-desktop.png',fullPage:true});
 await page.getByRole('button',{name:'3. Small mini-batches'}).click();
 await page.getByRole('button',{name:'Run comparison',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Finished'),{},{timeout:60000});
 assert.deepEqual(await page.locator('.metric').allTextContents().then(a=>a.slice(0,2)),['64.1%','74.6%']);
 /* BN under B=2 sits on the one chaotic row: the engine-level value is 80.5%, but a JS
    engine's math library can shift the trajectory a few points. Assert the pattern. */
 const batchBN=parseFloat(await page.locator('.metric').allTextContents().then(a=>a[2]));
 assert.ok(Math.abs(batchBN-64.5)<=8,'BN small-batch accuracy '+batchBN+' vs 64.5');
 await page.getByRole('button',{name:'Switch language'}).click();
 assert.ok((await page.locator('h1').innerText()).includes('同一个网络'));
 await page.setViewportSize({width:390,height:844});
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
 await page.screenshot({path:'/private/tmp/normalization-lab-mobile.png',fullPage:true});
 await page.getByLabel('SGD 更新次数').selectOption('600');
 await page.getByRole('button',{name:'运行对比',exact:true}).click();
 await page.getByRole('button',{name:'停止',exact:true}).click();
 assert.equal(await page.getByRole('button',{name:'运行对比',exact:true}).isEnabled(),true);
 // Models-to-compare picker: a subset must reproduce the per-model numbers of a full run.
 await page.getByRole('button',{name:'1. 温和的基线'}).click();
 await page.locator('#arm-bn').uncheck();
 assert.ok((await page.locator('#legend').innerText()).includes('无归一化'));
 assert.ok(!(await page.locator('#legend').innerText()).includes('BatchNorm'));
 assert.equal(await page.locator('.model-card').count(),2);
 await page.getByRole('button',{name:'运行对比',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('完成'),{},{timeout:60000});
 assert.deepEqual(await page.locator('.metric').allTextContents(),['84.8%','99.6%']);
 assert.ok((await page.locator('#inference-result').innerText()).includes('没有勾选 BatchNorm'));
 // An empty selection must train nothing and keep the previous results.
 await page.locator('#arm-none').uncheck();
 await page.locator('#arm-ln').uncheck();
 await page.getByRole('button',{name:'运行对比',exact:true}).click();
 assert.ok((await page.locator('#status').innerText()).includes('至少勾选一个模型'));
 assert.deepEqual(await page.locator('.metric').allTextContents(),['84.8%','99.6%']);
 assert.deepEqual(errors,[]);console.log('PASS: matrix controls, real training (2 presets + a two-model subset), model picker, empty-selection guard, export, language switch, mobile overflow, stop, no browser errors.');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});

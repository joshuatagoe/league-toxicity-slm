// Demo-video recorder for the League toxicity demo (the demo-video skill). Start demo/server.py first, then:
//   node demo/video/record.cjs   ->  raw.webm and waits.json beside this file
// (.cjs, so it runs as CommonJS even in a repo whose package.json says "type": "module".)
const { chromium } = require('playwright');
const fs = require('fs');

const BASE = process.env.BASE || 'http://127.0.0.1:8780';
const OUT = __dirname;
const SIZE = { width: 1280, height: 720 };

(async () => {
  fs.rmSync(OUT + '/raw', { recursive: true, force: true });
  // Warm the model on a single line first, before recording starts, so the first live line's timing isn't a cold start.
  await fetch(BASE + '/api/label', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ lines: ['warm up'] }) });
  const browser = await chromium.launch();
  // bypassCSP lets the overlays below draw on a strict-CSP page; it affects only this recording browser.
  const ctx = await browser.newContext({ viewport: SIZE, bypassCSP: true, recordVideo: { dir: OUT + '/raw', size: SIZE } });
  const p = await ctx.newPage();
  const t0 = Date.now();
  const now = () => (Date.now() - t0) / 1000;
  const waits = [];
  const scenes = [];

  const pause = (ms) => p.waitForTimeout(ms);

  // A full-screen numbered title card. Its title is the narration script's section heading, word for word.
  // `id` names the narration clip that starts here (00, 01, ...); cut.py writes each id's start to sections.json.
  async function caption(title, sub, ms = 3500, id = null) {
    scenes.push({ at: now(), title, id });
    await p.evaluate(([title, sub]) => {
      const el = document.createElement('div');
      el.id = 'vcap';
      el.style.cssText = 'position:fixed;inset:0;z-index:99999;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(17,20,24,.92);color:#fff;font:600 40px/1.25 system-ui,sans-serif;text-align:center;padding:48px';
      const h = document.createElement('div'); h.textContent = title; el.appendChild(h);
      if (sub) { const s = document.createElement('div'); s.textContent = sub; s.style.cssText = 'margin-top:18px;font:400 24px/1.4 system-ui,sans-serif;max-width:900px;color:#d7dde4'; el.appendChild(s); }
      document.body.appendChild(el);
    }, [title, sub]);
    await pause(ms);
    await p.evaluate(() => document.getElementById('vcap')?.remove());
  }

  // A small label bottom-left naming the scene that is playing. Survives until the next page load or banner.
  async function banner(text) {
    await p.evaluate((text) => {
      let el = document.getElementById('vbanner');
      if (!el) { el = document.createElement('div'); el.id = 'vbanner'; document.body.appendChild(el); }
      el.style.cssText = 'position:fixed;right:28px;top:88px;z-index:99998;padding:8px 14px;border-radius:8px;background:#111418;color:#fff;font:600 16px system-ui,sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.3)';
      el.textContent = text;
    }, text);
  }

  // Waits until the app is idle again, and logs the span so cut.py can speed it up. ADAPT the idle check to the app:
  // a spinner or "Thinking…" element going empty, a button re-enabling, or a network request finishing.
  async function waitReady() {
    const start = now();
    await pause(300);
    while (await p.evaluate(() => {
      const busy = document.querySelector('[aria-busy="true"], .spinner, #thinking:not(:empty)');
      return Boolean(busy);
    })) await pause(150);
    await pause(600); // let anything that renders just after the reply arrive
    waits.push([start + 1.2, now() - 0.4]); // keep a little of each wait at normal speed, so it reads as a wait
  }

  // Clicks a selector, or the first button whose text starts with `text` (a string) or matches it (a RegExp).
  async function click(target, { wait = true } = {}) {
    let el = null;
    if (typeof target === 'string' && /^[.#\[a-z]/i.test(target) && !target.includes(' ')) el = await p.$(target);
    if (!el) {
      for (const b of await p.$$('button, a, [role="button"]')) {
        const t = ((await b.textContent()) || '').trim();
        if (target instanceof RegExp ? target.test(t) : t.startsWith(target)) { el = b; break; }
      }
    }
    if (!el) { console.log('not found:', String(target)); return false; }
    await el.scrollIntoViewIfNeeded(); await el.hover(); await pause(700);
    await el.click();
    if (wait) await waitReady();
    return true;
  }

  async function scroll(px, ms = 4000) { await p.mouse.wheel(0, px); await pause(ms); }

  // ---- SCENES: one claim each. Titles match docs/walkthrough-script.md's headings word for word. ----
  // Hold lengths are sized to the narration (about 2.5 words a second; each section's words / 2.5 minus the 3.5 s card).
  async function press(el, wait) {  // click() below matches selectors without spaces only, so these pass the element
    await el.scrollIntoViewIfNeeded(); await el.hover(); await pause(700); await el.click();
    if (wait) await waitReady();
  }
  const tab = async (name) => { await press(await p.$(`nav button[data-tab="${name}"]`), false); await pause(400); };
  const set = async (i) => press((await p.$$('#sets .btn'))[i], true);
  async function typeLine(text) {
    await p.click('#line'); await pause(400);
    await p.type('#line', text, { delay: 70 }); await pause(500);
    await click('#go');
  }

  await p.goto(BASE + '/', { waitUntil: 'networkidle' });
  await p.waitForSelector('#sets .btn');
  await caption('League chat toxicity: a 1.7B model vs prompting', 'Qwen3-1.7B with a LoRA adapter, trained on real League chat labelled by experienced players.', 4500, '00');
  await banner('The demo page and the behaviour spec');
  await pause(4000);
  await p.hover('#chips'); await pause(8000);
  await p.hover('.spec'); await pause(21000);

  await caption('1. Prompting has a ceiling', 'Frontier models with their best prompts, on 1,000 real benchmark lines. The bar is precision 0.80.', 3500, '01');
  await tab('results');
  await banner('Scene 1: the best prompted models');
  await pause(3000);
  await p.hover('#resBody tr:nth-child(4)'); await pause(12000);
  await p.hover('#resBody tr:nth-child(5)'); await pause(11000);

  await caption('2. The untrained model misjudges real chat', 'Real benchmark lines, expert labels. Picked from the lines the untrained model gets wrong.', 3500, '02');
  await tab('try');
  await banner('Scene 2: untrained vs tuned on real chat');
  await set(0);
  await pause(9000);  // rows follow the narration: an insult let through, a report call, an apology, then "troll"
  await p.hover('#outBody tr:nth-child(2)'); await pause(3500);
  await p.hover('#outBody tr:nth-child(5)'); await pause(2500);
  await p.hover('#outBody tr:nth-child(7)'); await pause(1800);
  await p.hover('#outBody tr:nth-child(6)'); await pause(3000);
  await p.hover('#outTitle'); await pause(8000);

  await caption('3. Lines it has never seen, typed live', 'None of these lines appear in any training or eval file.', 3500, '03');
  await banner('Scene 3: typed live');
  await typeLine('go back to iron noob'); await pause(3500);
  await typeLine('report mid for afk'); await pause(3500);
  await typeLine('trash jungler'); await pause(8000);

  await caption('4. A data fix for what v1 got wrong', '676 teacher-written lines aimed at v1\u2019s failures, checked by a critic against the written rules.', 3500, '04');
  await banner('Scene 4: what the v2 data fixed');
  await set(1);
  await pause(4000);
  await p.hover('#outBody tr:nth-child(1)'); await pause(10000);
  await p.hover('#outBody tr:nth-child(3)'); await pause(15000);

  await caption('5. What it still misses', 'Hostility without insult words. Claude\u2019s best prompt catches all four.', 3500, '05');
  await banner('Scene 5: what v2 still misses');
  await set(2);
  await pause(4000);
  await p.hover('#outBody tr:nth-child(1)'); await pause(10000);
  await p.hover('#outBody tr:nth-child(4)'); await pause(10000);

  await caption('6. More expert data stops helping', 'v1 trained on 1,364 to 10,919 expert-labelled rows. F1 flattens after about 5,500 rows.', 3500, '06');
  await tab('results');
  await banner('Scene 6: the data-efficiency curve');
  await pause(3000);
  await p.hover('img.curve'); await pause(26000);

  await caption('7. The numbers, and how to check them', 'v2 clears the precision bar and misses the F1 bar. Public checkpoint, pinned commits, one command.', 3500, '07');
  await banner('Scene 7: results and the public checkpoint');
  await pause(2000);
  await p.hover('#resBody tr:nth-child(3)'); await pause(14000);
  await p.hover('#cmd'); await pause(17000);

  await caption('Thanks for watching', 'github.com/joshuatagoe/league-toxicity-slm \u00b7 huggingface.co/Juchen/league-toxicity-qwen3-1.7b', 7000, '08');
  // ---- end of scenes ----

  fs.writeFileSync(OUT + '/waits.json', JSON.stringify({ waits, scenes, total: now() }, null, 1));
  await ctx.close(); await browser.close();
  const file = fs.readdirSync(OUT + '/raw').find((f) => f.endsWith('.webm'));
  fs.renameSync(OUT + '/raw/' + file, OUT + '/raw.webm');
  console.log(`recorded ${now().toFixed(1)} s, ${waits.length} waits, ${scenes.length} title cards`);
})();

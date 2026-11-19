// Markdown paper -> print HTML -> PDF (bundled headless Chromium).
// usage: node build_pdf.js <paper.md> <out.pdf>
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { marked } = require('marked');  // npm install marked@12

const [md, out] = process.argv.slice(2);
const dir = path.dirname(path.resolve(md));
const src = fs.readFileSync(md, 'utf8');
const body = marked.parse(src);
const title = (src.match(/^# (.+)$/m) || [, 'paper'])[1].replace(/[<&]/g, '');
const html = `<!doctype html><html><head><meta charset="utf-8"><title>${title}</title><base href="file://${dir}/">
<style>
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
:root { --ink:#141414; --ink2:#55534e; --rule:#d9d6cf; --accent:#2a78d6; }
body { font-family: 'Noto Serif CJK TC','Noto Serif CJK JP','Noto Serif',serif; color:var(--ink);
       font-size: 10.5pt; line-height: 1.6; background:#fff; }
h1 { font-size: 17pt; line-height:1.35; text-align:center; margin: 0 0 6pt; }
h1 + p { text-align:center; color:var(--ink2); margin-top:0; }
h2 { font-size: 13pt; border-bottom: 1px solid var(--rule); padding-bottom: 2pt; margin-top: 18pt; }
h3 { font-size: 11.5pt; margin-top: 12pt; }
h1,h2,h3 { font-family: 'Noto Sans CJK TC','Noto Sans CJK JP','Noto Sans',sans-serif; break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; break-inside: avoid; }
th, td { border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f3f1ec; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; break-inside: avoid; }
p > em:only-child { display:block; font-size: 9pt; color: var(--ink2); text-align:center; }
blockquote { margin: 10pt 0; padding: 8pt 12pt; border-left: 4px solid var(--accent); background: #eef4fc; }
code { font-family: 'Noto Sans Mono CJK TC','DejaVu Sans Mono',monospace; font-size: 9pt; }
hr { border: 0; border-top: 1px solid var(--rule); }
p:has(+ table), p:has(+ ol), p:has(+ ul), p:has(> img) { break-after: avoid; }
table + p, p:has(> img) + p { break-before: avoid; }
li { break-inside: avoid; }
tr { break-inside: avoid; }
ol, ul { padding-left: 18pt; }
</style></head><body>${body}</body></html>`;
const tmp = path.join(dir, '.print.html');
fs.writeFileSync(tmp, html);
const chrome = process.env.CHROME || 'chromium';  // path to a Chrome/Chromium binary
execFileSync(chrome, ['--headless', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
  `--print-to-pdf=${path.resolve(out)}`, `file://${tmp}`], { stdio: 'ignore', timeout: 120000 });
fs.unlinkSync(tmp);
console.log('wrote', out);

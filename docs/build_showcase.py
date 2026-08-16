"""Regenerate the Ishigami showcase pages — one command, single source.

    python docs/build_showcase.py

Copies the figures referenced below from ``figures/`` into ``docs/figures/`` and
writes both ``docs/ishigami_showcase.md`` (renders on GitHub) and
``docs/ishigami_showcase.html`` (standalone, light/dark) from the SAME content
definition, so the two never drift.

Run ``python examples/run_ishigami_heteroscedastic.py`` first to (re)produce the
figures in ``figures/``.
"""

import os
import re
import shutil
import html as _html

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_FIG = os.path.join(HERE, os.pardir, 'figures')   # repo figures/ (git-ignored)
DST_FIG = os.path.join(HERE, 'figures')              # committed docs/figures/


# ---------------------------------------------------------------------------
# Content — defined once, rendered to both Markdown and HTML.
# Captions use **bold**, *italic*, `code`, and [text](url); plain Unicode is fine.
# ---------------------------------------------------------------------------
def F(file, alt, cap):
    return {'file': file, 'alt': alt, 'cap': cap}


SECTIONS = [
    {'num': 1,
     'title': 'The dual spectrum — mean sensitivity vs log-variance index S^h',
     'body': [
        ('p', 'For the fixed fitted configuration, the conditional-mean '
              'sensitivity and fitted log-variance index are read from the '
              'fitted coefficients — one pair of numbers per variable.'),
        ('fig', F('ishigami_sensitivity_glyphs.png',
                  'Dual-sensitivity ellipse glyphs per variable',
                  '**Dual-sensitivity glyphs.** Each variable is one ellipse: '
                  '**width ∝ its effect on the mean** E[y|x], and **height '
                  '∝ its fitted log-variance index** S^h. x₁/x₂ '
                  'are wide and flat (mean drivers); **x₃ is tall and '
                  'narrow** — it barely touches the mean yet drives the '
                  'fitted multiplicative residual scale, the hidden-driver signature. The shape '
                  'tells the whole story at a glance.')),
     ]},
    {'num': 2,
     'title': 'The learned first-order effects',
     'body': [
        ('p', 'The fitted one-dimensional effect of each variable, in quantile '
              'space.'),
        ('fig', F('ishigami_components.png',
                  'Learned first-order component functions',
                  'x₁ recovers a sine, x₂ a squared-sine (two humps). '
                  '**x₃ is exactly flat** — its spurious main-effect '
                  "block was removed by `first_order_pruning='bic'`, a "
                  'model-selection heuristic that can zero an entire first-order '
                  'block. It is not a calibrated significance test. Plain ridge '
                  'can only shrink such a block; it can '
                  'never set it to zero.')),
     ]},
    {'num': 3,
     'title': 'Regularization — choosing and understanding the penalty',
     'body': [
        ('p', 'Every quantity below comes from a single ridge factorization, so '
              'an entire regularization path is essentially free — no '
              'retraining.'),
        ('fig', F('ishigami_reg_path.png', 'Four-panel regularization path',
                  '**Regularization path.** Clockwise: the L-curve (fit vs '
                  'complexity) with the GCV optimum starred; GCV & evidence '
                  'agreeing on λ; the Sobol indices at every λ (note '
                  'x₃ pinned at zero throughout); and the explained-variance '
                  'decomposition by interaction order.')),
        ('fig', F('ishigami_pareto.png',
                  'Pareto frontier: complexity vs unexplained variance',
                  '**Pareto frontier.** Unexplained variance vs model complexity '
                  '(effective degrees of freedom), colored by λ. The GCV '
                  'optimum sits at the elbow — where extra complexity stops '
                  'buying accuracy.')),
     ]},
    {'num': 4,
     'title': 'How good is the fit — under noise?',
     'body': [
        ('p', 'With heteroscedastic noise, a prediction-vs-observation plot '
              '*cannot* collapse to a line: its scatter is the irreducible '
              'noise. Because the data is synthetic we also know the noiseless '
              'truth, so we can separate **mean recovery** from **noise**.'),
        ('row',
         F('ishigami_parity_observed.png', 'Predicted vs observed parity',
           'Predicted vs **observed** y — colored by the true noise std '
           'σ(x). R² = 0.79, at the noise ceiling (not a weak fit); the '
           'points that stray farthest are the high-noise ones.'),
         F('ishigami_parity_truth.png', 'Predicted vs true function parity',
           'Predicted vs the **true** f(x) — a tight line, R² = 0.98. '
           'The mean is recovered well; the gap on the left is purely noise.')),
        ('fig', F('ishigami_intervals.png',
                  'Prediction intervals widen with x3',
                  '**Prediction intervals from the mean + log-variance fit.** A 1-D '
                  'slice (x₁ = 0, x₂ = π/2) so the mean is flat and '
                  'only the noise changes with x₃. The 95% band = mean ± '
                  '2σ̂(x) from both models together; it widens with '
                  'x₃ and tracks the true ±2σ (green dotted), '
                  'covering the observed points.')),
        ('row',
         F('ishigami_surface.png', 'Transparent true vs fitted surface',
           '**True vs fitted mean surface** (slice x₃ = 0). The true surface '
           '(blue) and the predicted mean (orange) overlap so closely they '
           'blend.'),
         F('ishigami_variance_fit.png', 'Predicted vs true noise std',
           '**Fitted residual-scale recovery.** Predicted residual std σ̂(x) '
           'vs the synthetic true σ(x) — correlation 0.99 in this in-sample '
           'diagnostic.')),
     ]},
    {'num': 5,
     'title': 'Verifying the fit',
     'body': [
        ('p', 'A single call, `verify_model(...)`, runs the diagnostic workflow '
              'end-to-end and confirms the fit is internally consistent before '
              'its Sobol indices are trusted:'),
        ('code',
         "[PASS] Sobol additivity: first+second+third+residual = 1.000 (target 1.000)\n"
         "[PASS] Sobol bounds: indices in [0,1], total-order >= first-order\n"
         "[PASS] Fit quality (test R^2): R^2 = 0.802\n"
         "[PASS] Calibration (coverage): cov90=0.89 cov95=0.95 var(z)=1.07\n"
         "[PASS] Input correlation: correlation_level = 'clean'\n"
         "[info] Pure-interaction variables: x3: zero first-order, nonzero total-order\n"
         "  => ALL CHECKS PASSED"),
        ('p', 'The `[info]` line is the payoff: the toolbox has correctly '
              'identified x₃ as a pure-interaction, hidden-variance variable '
              '— exactly the structure a single feature-importance number '
              'would miss.'),
     ]},
]


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------
def render_md():
    p = []
    p.append('# HiFi-ANOVA — a worked example on the heteroscedastic '
             'Ishigami function\n')
    p.append('*A visual tour of what the toolbox produces from one fit: the dual '
             'mean/log-variance spectrum, the learned effects, the '
             'regularization trade-offs, and how the fit holds up under noise.*\n')
    p.append('> There is also a standalone [HTML version](ishigami_showcase.html) '
             'of this page (light/dark, same content).\n')
    p.append('The **Ishigami function** is a classic sensitivity-analysis '
             'benchmark. We use a **heteroscedastic** variant — the response '
             'is noisy, and the noise level itself depends on an input — '
             'because it exercises every part of the toolbox at once.\n')
    p.append('> **f(x) = sin(x₁) + 7·sin²(x₂) + '
             '0.1·x₃⁴·sin(x₁)**,  xᵢ ~ '
             'U(−π, π), with Gaussian noise whose std ramps '
             '**0.3 → 3.0** across x₃.\n')
    p.append('**Why this example is instructive — two things about x₃:**\n')
    p.append('- It has **zero first-order effect** on the mean (it acts only '
             'through the x₁–x₃ interaction) but a **non-zero '
             'total-order** effect. A method that reports a first-order '
             'importance for x₃ is picking up noise.\n'
             '- It is a **hidden log-variance driver**: it carries no mean signal yet '
             'controls the fitted multiplicative residual scale — invisible to ordinary '
             'feature importance.\n')
    p.append('Analytic ground-truth first-order Sobol indices: **x₁ = 0.314, '
             'x₂ = 0.442, x₃ = 0.000**; x₃ total-order = 0.244.\n')
    p.append('Everything below is produced by '
             '`python examples/run_ishigami_heteroscedastic.py` (7 000-point '
             'fit; all figures land in `figures/`).\n')
    p.append('---')

    for s in SECTIONS:
        p.append(f"\n## {s['num']}. {s['title']}\n")
        for item in s['body']:
            if item[0] == 'p':
                p.append(item[1] + '\n')
            elif item[0] == 'code':
                p.append('```\n' + item[1] + '\n```\n')
            elif item[0] == 'fig':
                f = item[1]
                p.append(f"![{f['alt']}](figures/{f['file']})\n")
                # Plain caption (bold lead already marks it; avoids italic/bold
                # delimiter clashes and matches the HTML figcaption styling).
                p.append(f"{f['cap']}\n")
            elif item[0] == 'row':
                a, b = item[1], item[2]
                p.append(f"| ![{a['alt']}](figures/{a['file']}) | "
                         f"![{b['alt']}](figures/{b['file']}) |\n"
                         f"|:--|:--|\n"
                         f"| {a['cap']} | {b['cap']} |\n")
        p.append('---')

    p.append('\nReproduce: `python examples/run_ishigami_heteroscedastic.py`. See '
             'the [User Guide](USER_GUIDE.md) for the full option reference and '
             'the [benchmark](../benchmarks/README.md) to compare your own '
             'model.\n')
    p.append('*Documentation notice: Copyright (c) 2026 R. Sala. Draft, work in '
             'progress — not covered by the source-code license. See '
             '[LICENSING.md](../LICENSING.md).*')
    return "\n".join(p) + "\n"


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------
def _inline(s):
    s = _html.escape(s, quote=False)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', s)
    s = re.sub(r'`(.+?)`', r'<code>\1</code>', s)
    s = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', s)
    return s


def _figure(f):
    return (f'<figure><img loading="lazy" alt="{_html.escape(f["alt"])}" '
            f'src="figures/{f["file"]}"><figcaption>{_inline(f["cap"])}'
            f'</figcaption></figure>')


CSS = """
:root{--bg:#fff;--fg:#1a1d24;--muted:#5c6470;--card:#f7f8fa;--border:#e4e7ec;
--accent:#3274A1;--code:#f0f2f5;--maxw:940px}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e6e9ef;--muted:#9aa4b2;
--card:#1c2027;--border:#2b313b;--accent:#7fb2d8;--code:#1a1e25}}
:root[data-theme="light"]{--bg:#fff;--fg:#1a1d24;--muted:#5c6470;--card:#f7f8fa;
--border:#e4e7ec;--accent:#3274A1;--code:#f0f2f5}
:root[data-theme="dark"]{--bg:#14171c;--fg:#e6e9ef;--muted:#9aa4b2;--card:#1c2027;
--border:#2b313b;--accent:#7fb2d8;--code:#1a1e25}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
line-height:1.6;font-size:16px}
.wrap{max-width:var(--maxw);margin:0 auto;padding:2.2rem 1.2rem 4rem}
header{border-bottom:1px solid var(--border);padding-bottom:1.4rem;margin-bottom:1.6rem}
h1{font-size:1.9rem;line-height:1.25;margin:.2rem 0 .4rem}
h2{font-size:1.35rem;margin:2.6rem 0 .3rem;padding-top:.6rem}
h2 .n{color:var(--accent);font-variant-numeric:tabular-nums;margin-right:.5rem}
.lede{color:var(--muted);font-size:1.05rem}
p{margin:.7rem 0}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.formula{background:var(--card);border:1px solid var(--border);border-radius:10px;
padding:1rem 1.1rem;margin:1.1rem 0;font-size:1.05rem}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
background:var(--code);border:1px solid var(--border);border-radius:5px;
padding:.08em .4em;font-size:.9em}
pre{background:var(--code);border:1px solid var(--border);border-radius:10px;
padding:1rem 1.1rem;overflow-x:auto;font-size:.86rem;line-height:1.5}
pre code{background:none;border:none;padding:0}
figure{margin:1.4rem 0;background:var(--card);border:1px solid var(--border);
border-radius:12px;padding:1rem;overflow:hidden}
figure img{display:block;width:100%;height:auto;max-width:100%;border-radius:6px;background:#fff}
figcaption{color:var(--muted);font-size:.92rem;margin-top:.7rem}
figcaption b{color:var(--fg)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.grid figure{margin:0}
@media (max-width:720px){.grid{grid-template-columns:1fr}}
.key{background:var(--card);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;
padding:.7rem 1rem;margin:1.1rem 0}
.muted{color:var(--muted)}
.toggle{position:fixed;top:1rem;right:1rem;background:var(--card);
border:1px solid var(--border);color:var(--fg);border-radius:8px;padding:.35rem .7rem;
cursor:pointer;font-size:.85rem}
footer{border-top:1px solid var(--border);margin-top:3rem;padding-top:1.2rem;
color:var(--muted);font-size:.85rem}
"""

JS = ("const b=document.querySelector('.toggle');b.addEventListener('click',()=>{"
      "const r=document.documentElement;const c=r.getAttribute('data-theme')||"
      "(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');"
      "r.setAttribute('data-theme',c==='dark'?'light':'dark');});")


def render_html():
    p = ['<div class="wrap">',
         '<button class="toggle" title="toggle light/dark">◐ theme</button>',
         '<header>',
         '<h1>HiFi-ANOVA — a worked example on the heteroscedastic Ishigami '
         'function</h1>',
         '<p class="lede">A visual tour of what the toolbox produces from one '
         'fit: the dual mean/log-variance spectrum, the learned effects, '
         'the regularization trade-offs, and how the fit holds up under noise.</p>',
         '</header>',
         '<p>The <b>Ishigami function</b> is a classic sensitivity-analysis '
         'benchmark. We use a <b>heteroscedastic</b> variant — the response '
         'is noisy, and the noise level itself depends on an input — because '
         'it exercises every part of the toolbox at once.</p>',
         '<div class="formula"><code>f(x) = sin(x₁) + 7·sin²(x₂) '
         '+ 0.1·x₃⁴·sin(x₁)</code>, &nbsp; xᵢ ~ '
         'U(−π, π), &nbsp; with Gaussian noise whose std ramps '
         '<b>0.3 → 3.0</b> across x₃.</div>',
         '<div class="key">Why this example is instructive — two things about '
         '<b>x₃</b>:<ul>'
         '<li>It has <b>zero first-order effect</b> on the mean (it acts only '
         'through the x₁–x₃ interaction) but a <b>non-zero '
         'total-order</b> effect. A method that reports a first-order importance '
         'for x₃ is picking up noise.</li>'
         '<li>It is a <b>hidden log-variance driver</b>: it carries no mean signal yet '
         'controls the fitted multiplicative residual scale — invisible to ordinary feature '
         'importance.</li></ul>'
         'Analytic ground-truth first-order Sobol indices: <b>x₁ = 0.314, '
         'x₂ = 0.442, x₃ = 0.000</b>; x₃ total-order = 0.244.</div>',
         '<p class="muted">Everything below is produced by '
         '<code>python examples/run_ishigami_heteroscedastic.py</code> '
         '(7 000-point fit; all figures land in <code>figures/</code>).</p>']

    for s in SECTIONS:
        p.append(f'<h2><span class="n">{s["num"]}</span>'
                 f'{_html.escape(s["title"])}</h2>')
        for item in s['body']:
            if item[0] == 'p':
                p.append(f'<p>{_inline(item[1])}</p>')
            elif item[0] == 'code':
                p.append(f'<pre><code>{_html.escape(item[1])}</code></pre>')
            elif item[0] == 'fig':
                p.append(_figure(item[1]))
            elif item[0] == 'row':
                p.append('<div class="grid">' + _figure(item[1])
                         + _figure(item[2]) + '</div>')

    p.append('<footer><p>Reproduce: '
             '<code>python examples/run_ishigami_heteroscedastic.py</code>. See '
             'the <a href="USER_GUIDE.md">User Guide</a> for the full option '
             'reference and the <a href="../benchmarks/README.md">benchmark</a> '
             'to compare your own model.</p>'
             '<p>Documentation notice: Copyright (c) 2026 R. Sala. Draft, work in '
             'progress — not covered by the source-code license. See '
             'LICENSING.md.</p></footer></div>')

    body = "\n".join(p)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>HiFi-ANOVA — Ishigami worked example</title>'
            f'<style>{CSS}</style></head><body>{body}'
            f'<script>{JS}</script></body></html>\n')


def main():
    # Collect referenced figures and copy them into docs/figures/.
    files = []
    for s in SECTIONS:
        for item in s['body']:
            if item[0] == 'fig':
                files.append(item[1]['file'])
            elif item[0] == 'row':
                files += [item[1]['file'], item[2]['file']]
    os.makedirs(DST_FIG, exist_ok=True)
    missing = []
    for f in files:
        src = os.path.join(SRC_FIG, f)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(DST_FIG, f))
        else:
            missing.append(f)
    if missing:
        raise SystemExit("Missing figures in figures/ (run the example first): "
                         + ", ".join(missing))

    with open(os.path.join(HERE, 'ishigami_showcase.md'), 'w') as fh:
        fh.write(render_md())
    with open(os.path.join(HERE, 'ishigami_showcase.html'), 'w') as fh:
        fh.write(render_html())
    print(f"Wrote docs/ishigami_showcase.md and .html; copied {len(files)} "
          f"figures into docs/figures/.")


if __name__ == '__main__':
    main()

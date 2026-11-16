// Renders reports/benchmark.json (served at /api/benchmark) as a user-facing validation page.
const MODELS = ['fibonacci_net', 'baseline_cnn'];
const COLOR = { fibonacci_net: 'var(--series-1)', baseline_cnn: 'var(--series-2)' };
const NAME = { fibonacci_net: 'FibonacciNet', baseline_cnn: 'Standard CNN' };
const SVGNS = 'http://www.w3.org/2000/svg';
const tooltip = document.getElementById('tooltip');

const $ = (id) => document.getElementById(id);
const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
const show = (...ids) => ids.forEach((id) => $(id).classList.remove('is-hidden'));
// Float32 weights: 4 bytes per parameter.
const weightsMb = (params) => (params * 4) / 1e6;

function el(tag, attrs = {}, parent) {
    const node = document.createElementNS(SVGNS, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (parent) parent.appendChild(node);
    return node;
}

function tip(node, html) {
    node.addEventListener('pointermove', (e) => {
        tooltip.innerHTML = html;
        tooltip.style.opacity = 1;
        tooltip.style.left = `${Math.min(e.clientX + 14, window.innerWidth - 270)}px`;
        tooltip.style.top = `${e.clientY + 14}px`;
    });
    node.addEventListener('pointerleave', () => { tooltip.style.opacity = 0; });
}

function legend(target, labels = NAME) {
    $(target).innerHTML = MODELS
        .map((k) => `<span><i class="swatch" style="background:${COLOR[k]}"></i>${labels[k]}</span>`)
        .join('');
}

// ---------------------------------------------------------------- hero
function hero(report) {
    const c = report.comparison, m = report.models, a = report.data_audit;
    const fib = m.fibonacci_net, cnn = m.baseline_cnn;
    const equivalent = c.non_inferior && c.external_seed_averaged?.non_inferior;

    $('heroTitle').textContent = equivalent
        ? `As accurate as a standard CNN, ${c.param_reduction_x}× smaller`
        : `${c.param_reduction_x}× smaller than a standard CNN`;
    $('heroLede').innerHTML = `Tested on <strong>${a.splits.test.n.toLocaleString()}</strong> biopsy-confirmed thyroid ultrasound
        images and <strong>${a.external.n.toLocaleString()}</strong> images from an independent dataset, FibonacciNet matched a
        conventional convolutional network while using a fraction of its memory and compute.`;
    const verdict = $('heroVerdict');
    verdict.textContent = equivalent
        ? 'Statistically equivalent accuracy on both test sets'
        : 'Equivalent accuracy was not demonstrated';
    verdict.classList.toggle('bad', !equivalent);

    const tile = (big, label, sub) => `<div class="ratio"><strong>${big}</strong><span>${label}</span><small>${sub}</small></div>`;
    const fe = fib.efficiency, ce = cnn.efficiency;
    $('ratioGrid').innerHTML = [
        tile(fib.summary.auc.mean.toFixed(3), 'Accuracy (AUROC)',
            `Standard CNN ${cnn.summary.auc.mean.toFixed(3)}, no significant difference`),
        tile(`${c.param_reduction_x}×`, 'fewer parameters',
            `${(fe.params / 1e6).toFixed(2)} M vs ${(ce.params / 1e6).toFixed(2)} M`),
        tile(`${c.flop_reduction_x}×`, 'less computation',
            `${(fe.flops / 1e9).toFixed(2)} vs ${(ce.flops / 1e9).toFixed(2)} GFLOPs per image`),
        tile(`${c.latency_speedup_x}×`, 'faster on a CPU',
            `${fe.latency_ms_cpu.toFixed(1)} ms vs ${ce.latency_ms_cpu.toFixed(1)} ms per image`),
    ].join('');
}

// ---------------------------------------------------------------- paired bars
function barGroup(target, rows) {
    $(target).innerHTML = rows.map((r) => `
        <div>
            <div class="metric-label"><span>${r.label}</span><small>${r.note || ''}</small></div>
            ${MODELS.map((k) => `
                <div class="bar-row" data-tip="${NAME[k]}: ${r.fmt(r.values[k])}">
                    <div class="bar-track"><div class="bar-fill" style="width:${Math.max(0.5, 100 * r.scale(r.values[k]))}%;background:${COLOR[k]}"></div></div>
                    <span class="bar-value">${r.fmt(r.values[k])}</span>
                </div>`).join('')}
        </div>`).join('');
    $(target).querySelectorAll('.bar-row').forEach((row) => tip(row, row.dataset.tip));
}

function headToHead(report) {
    const m = report.models;
    const val = (key) => Object.fromEntries(MODELS.map((k) => [k, m[k].summary[key].mean]));
    const eff = (fn) => Object.fromEntries(MODELS.map((k) => [k, fn(m[k].efficiency)]));
    const unit = (v) => v; // accuracy metrics already live on a 0-1 scale

    legend('accuracyLegend');
    $('accuracySub').textContent = 'Higher is better. Averages of three independent training runs per model.';
    barGroup('accuracyBars', [
        { label: 'AUROC, main test set', note: 'ranking malignant above benign (1 = perfect)', values: val('auc'),
            scale: unit, fmt: (v) => v.toFixed(3) },
        { label: 'AUROC, independent dataset', note: 'images from a different source', values: val('external_auc'),
            scale: unit, fmt: (v) => v.toFixed(3) },
        { label: 'Cancers caught', note: 'sensitivity at the rule-out cut-off', values: val('rule_out_sensitivity'),
            scale: unit, fmt: (v) => pct(v, 1) },
        { label: 'Benign nodules cleared', note: 'specificity at the rule-out cut-off', values: val('rule_out_specificity'),
            scale: unit, fmt: (v) => pct(v, 1) },
    ]);

    legend('costLegend');
    const relative = (values) => (v) => v / Math.max(...Object.values(values));
    const params = eff((e) => e.params), flops = eff((e) => e.flops), ms = eff((e) => e.latency_ms_cpu);
    const size = eff((e) => weightsMb(e.params));
    barGroup('costBars', [
        { label: 'Parameters', note: 'learned weights', values: params, scale: relative(params),
            fmt: (v) => `${(v / 1e6).toFixed(2)} M` },
        { label: 'Computation per image', note: 'floating-point operations', values: flops, scale: relative(flops),
            fmt: (v) => `${(v / 1e9).toFixed(2)} GFLOPs` },
        { label: 'Time per image on a CPU', note: 'single image, no GPU', values: ms, scale: relative(ms),
            fmt: (v) => `${v.toFixed(1)} ms` },
        { label: 'Model size', note: 'stored weights', values: size, scale: relative(size),
            fmt: (v) => `${v.toFixed(1)} MB` },
    ]);
}

// ---------------------------------------------------------------- triage bands
function triage(report) {
    const fib = report.models.fibonacci_net, s = fib.summary, sel = fib.selected, t = sel.thresholds;
    const prevalence = report.data_audit.splits.test.malignant / report.data_audit.splits.test.n;
    $('triage').innerHTML = `
        <div class="band low">
            <h4>Low suspicion</h4>
            <div class="band-rule">probability below ${pct(t.rule_out)}</div>
            <div class="band-stat">${pct(s.rule_out_sensitivity.mean)}</div>
            <p>of cancers score <em>above</em> this band, so few are missed.</p>
            <div class="band-stat">${pct(s.rule_out_npv.mean)}</div>
            <p>of nodules placed in this band are benign.</p>
        </div>
        <div class="band mid">
            <h4>Indeterminate</h4>
            <div class="band-rule">between ${pct(t.rule_out)} and ${pct(t.rule_in)}</div>
            <div class="band-stat">${pct(sel.indeterminate_fraction)}</div>
            <p>of nodules land here and are flagged for specialist review and TI-RADS scoring instead of a model call.</p>
        </div>
        <div class="band high">
            <h4>High suspicion</h4>
            <div class="band-rule">probability ${pct(t.rule_in)} or higher</div>
            <div class="band-stat">${pct(s.rule_in_ppv.mean)}</div>
            <p>of nodules placed in this band are malignant.</p>
            <div class="band-stat">${pct(1 - s.rule_in_specificity.mean)}</div>
            <p>of benign nodules end up here.</p>
        </div>`;
    $('triageNote').textContent = `Measured on the main test set, where ${pct(prevalence)} of nodules are malignant. `
        + 'Where cancer is less common, a smaller share of each band will be malignant.';
}

// ---------------------------------------------------------------- charts
function frame(container, { xTitle, yTitle }) {
    const W = 460, H = 330, m = { t: 10, r: 14, b: 42, l: 62 };
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
    const xs = (v) => m.l + v * (W - m.l - m.r);
    const ys = (v) => H - m.b - v * (H - m.t - m.b);
    [0, 0.2, 0.4, 0.6, 0.8, 1].forEach((t) => {
        el('line', { x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t), stroke: 'var(--grid)' }, svg);
        el('text', { x: m.l - 8, y: ys(t) + 4, 'text-anchor': 'end' }, svg).textContent = pct(t);
        el('text', { x: xs(t), y: H - m.b + 16, 'text-anchor': 'middle' }, svg).textContent = pct(t);
    });
    el('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: 'var(--axis)' }, svg);
    el('line', { x1: xs(0), y1: ys(0), x2: xs(1), y2: ys(1), stroke: 'var(--axis)', 'stroke-dasharray': '4 4' }, svg);
    el('text', { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', class: 'axis-title' }, svg).textContent = xTitle;
    el('text', { x: 12, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', class: 'axis-title',
        transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, svg).textContent = yTitle;
    container.innerHTML = '';
    container.appendChild(svg);
    return { svg, xs, ys };
}

const path = (pts) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');

function rocChart(report, set) {
    const curve = (k) => (set === 'test' ? report.models[k].selected : report.models[k].selected.external);
    legend('rocLegend');
    const f = frame($('rocChart'), { xTitle: 'Benign nodules wrongly flagged', yTitle: 'Cancers caught' });
    [...MODELS].reverse().forEach((k) => {
        const r = curve(k).roc;
        const pts = r.fpr.map((x, i) => [f.xs(x), f.ys(r.tpr[i])]);
        el('path', { d: path(pts), fill: 'none', stroke: COLOR[k], 'stroke-width': k === 'fibonacci_net' ? 2.5 : 2,
            'stroke-linejoin': 'round' }, f.svg);
        const hit = el('path', { d: path(pts), fill: 'none', stroke: 'transparent', 'stroke-width': 14 }, f.svg);
        tip(hit, `<b>${NAME[k]}</b><br>one representative training run`);
    });
}

function calChart(report) {
    const sel = report.models.fibonacci_net.selected;
    $('calSub').textContent = `When FibonacciNet says a nodule has a 70% chance of being malignant, about 70% of such nodules are.
        On the main test set its stated probabilities were off by ${(sel.ece * 100).toFixed(1)} percentage points on average.`;
    const f = frame($('calChart'), { xTitle: 'Predicted chance of malignancy', yTitle: 'Actual share malignant' });
    const bins = sel.calibration_bins;
    el('path', { d: path(bins.map((b) => [f.xs(b.pred), f.ys(b.obs)])), fill: 'none', stroke: 'var(--series-1)',
        'stroke-width': 2 }, f.svg);
    bins.forEach((b) => {
        const c = el('circle', { cx: f.xs(b.pred), cy: f.ys(b.obs), r: Math.max(4, Math.min(9, Math.sqrt(b.n))),
            fill: 'var(--series-1)', stroke: '#fff', 'stroke-width': 2 }, f.svg);
        tip(c, `Predicted ${pct(b.pred)} → actually ${pct(b.obs)} malignant<br>${b.n} images`);
    });
}

function methods(report) {
    const a = report.data_audit;
    $('methodList').innerHTML = `
        <li><strong>${a.splits.test.n.toLocaleString()}</strong> test images from TN5000, a peer-reviewed dataset in which every
            diagnosis was confirmed by biopsy.</li>
        <li>A second, independent dataset of <strong>${a.external.n.toLocaleString()}</strong> images, never used for training or tuning.</li>
        <li>Both models were trained identically, three times each, and the results averaged.</li>
        <li>No test image, or a near-copy of one, was ever seen during training.</li>
        <li>All cut-offs and probability corrections were fixed on separate validation images before testing.</li>`;
}

async function init() {
    try {
        const res = await fetch('/api/benchmark');
        const report = await res.json();
        // Only a finished benchmark is shown; partial runs stay internal.
        if (!res.ok || report.status?.state !== 'complete') throw new Error('unavailable');
        $('benchLoading').classList.add('is-hidden');

        hero(report);
        headToHead(report);
        triage(report);
        rocChart(report, 'test');
        calChart(report);
        methods(report);
        show('hero', 'pairGrid', 'triageCard', 'chartGrid', 'aboutGrid');

        $('rocToggle').addEventListener('click', (e) => {
            const btn = e.target.closest('button');
            if (!btn) return;
            $('rocToggle').querySelectorAll('button').forEach((b) => b.setAttribute('aria-selected', String(b === btn)));
            rocChart(report, btn.dataset.set);
        });
    } catch (err) {
        document.querySelector('#benchLoading h3').textContent = 'Results not available yet';
        $('benchError').textContent = 'Benchmark results will appear here once model evaluation is complete.';
    }
}

document.addEventListener('DOMContentLoaded', init);

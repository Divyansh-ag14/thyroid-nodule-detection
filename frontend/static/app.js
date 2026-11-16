document.addEventListener('DOMContentLoaded', () => {
    const $ = (id) => document.getElementById(id);
    const fileInput = $('fileInput');
    const dropZone = $('dropZone');
    const views = { intake: [$('intake'), $('facts')], working: [$('working')], result: [$('result')] };

    let currentFile = null;
    let currentSample = null;
    let previewUrl = null;

    const pct = (v) => `${Math.round(v * 100)}%`;

    function showView(name) {
        Object.entries(views).forEach(([key, els]) => els.forEach((el) => el.classList.toggle('is-hidden', key !== name)));
        const hasResult = name === 'result';
        $('newCaseBtn').classList.toggle('is-hidden', !hasResult);
        $('reportBtn').classList.toggle('is-hidden', !hasResult);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function showAlert(message, type = 'error') {
        const alert = $('customAlert');
        alert.textContent = message;
        alert.className = `alert is-${type}`;
        alert.style.display = 'block';
        clearTimeout(showAlert.timer);
        showAlert.timer = setTimeout(() => { alert.style.display = 'none'; }, 4500);
    }

    async function errorMessage(response, fallback) {
        try {
            return (await response.json()).error || fallback;
        } catch {
            return fallback;
        }
    }

    // ---------------------------------------------------------------- samples
    async function loadSamples() {
        try {
            const data = await (await fetch('/static/samples/samples.json')).json();
            $('sampleAttribution').textContent = `Sample images: ${data.attribution}.`;
            $('sampleGrid').innerHTML = data.samples.map((s, i) => `
                <button class="sample" type="button" data-index="${i}">
                    <img src="${s.file}" alt="${s.title} ultrasound image" loading="lazy">
                    <span>${s.title}</span>
                </button>`).join('');
            $('sampleGrid').addEventListener('click', async (e) => {
                const btn = e.target.closest('.sample');
                if (!btn) return;
                const sample = data.samples[Number(btn.dataset.index)];
                const blob = await (await fetch(sample.file)).blob();
                analyse(new File([blob], `${sample.title}.jpg`, { type: blob.type || 'image/jpeg' }), sample);
            });
        } catch {
            document.querySelector('.samples').classList.add('is-hidden');
        }
    }

    // ---------------------------------------------------------------- analysis
    async function analyse(file, sample = null) {
        if (!file || !file.type.startsWith('image/')) {
            showAlert('Please choose a PNG or JPEG ultrasound image.');
            return;
        }
        currentFile = file;
        currentSample = sample;
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = URL.createObjectURL(file);
        $('workingPreview').src = previewUrl;
        $('workingName').textContent = file.name;
        showView('working');

        const form = new FormData();
        form.append('file', file);
        try {
            const response = await fetch('/analyze', { method: 'POST', body: form });
            if (!response.ok) throw new Error(await errorMessage(response, 'The image could not be analysed.'));
            renderResult(await response.json());
            showView('result');
        } catch (err) {
            showView('intake');
            showAlert(err.message);
        }
    }

    function renderResult(r) {
        const t = r.thresholds;
        const ruleIn = t.rule_in ?? t.balanced;

        $('resultFile').textContent = currentFile.name;
        $('riskPanel').className = `risk ${r.risk_band}`;
        $('bandChip').textContent = r.risk_label;
        $('probValue').textContent = pct(r.probability);
        $('guidance').textContent = r.guidance;

        // Risk scale: segment widths follow the validation-fixed cut-offs.
        const [low, mid, high] = $('scaleTrack').querySelectorAll('.seg');
        low.style.width = `${t.rule_out * 100}%`;
        mid.style.width = `${(ruleIn - t.rule_out) * 100}%`;
        high.style.width = `${(1 - ruleIn) * 100}%`;
        $('scaleMarker').style.left = `${Math.min(99.5, Math.max(0.5, r.probability * 100))}%`;
        $('scaleLabels').innerHTML = `
            <span style="left:${t.rule_out * 100}%">${pct(t.rule_out)}<br>rule-out</span>
            <span style="left:${ruleIn * 100}%">${pct(ruleIn)}<br>rule-in</span>`;

        $('inputImg').src = `data:image/png;base64,${r.original_image}`;
        if (r.gradcam_image) {
            $('camImg').src = `data:image/png;base64,${r.gradcam_image}`;
            $('camImg').parentElement.classList.remove('is-hidden');
            // Grad-CAM explains the side of the 0.5 decision boundary the raw score fell on.
            $('camCaption').textContent = `Evidence map: warm areas pushed the model towards ${r.score >= 0.5 ? 'malignant' : 'benign'}.`;
        } else {
            $('camImg').parentElement.classList.add('is-hidden');
        }

        const truth = $('truthChip');
        truth.classList.toggle('is-hidden', !currentSample);
        if (currentSample) {
            truth.textContent = `Biopsy result: ${currentSample.biopsy}`;
        }
    }

    // ---------------------------------------------------------------- report
    async function downloadReport() {
        if (!currentFile) return;
        const btn = $('reportBtn');
        btn.disabled = true;
        btn.textContent = 'Preparing report…';
        const form = new FormData();
        form.append('file', currentFile);
        try {
            const response = await fetch('/report', { method: 'POST', body: form });
            if (!response.ok) throw new Error(await errorMessage(response, 'Report generation failed.'));
            const url = URL.createObjectURL(await response.blob());
            const a = Object.assign(document.createElement('a'), { href: url, download: `thyroid-report-${Date.now()}.docx` });
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
            showAlert('Report downloaded.', 'success');
        } catch (err) {
            showAlert(err.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Download report';
        }
    }

    function newCase() {
        currentFile = null;
        currentSample = null;
        fileInput.value = '';
        showView('intake');
    }

    // ---------------------------------------------------------------- wiring
    ['dragenter', 'dragover'].forEach((ev) => dropZone.addEventListener(ev, (e) => {
        e.preventDefault();
        dropZone.classList.add('active');
    }));
    ['dragleave', 'drop'].forEach((ev) => dropZone.addEventListener(ev, (e) => {
        e.preventDefault();
        dropZone.classList.remove('active');
    }));
    dropZone.addEventListener('drop', (e) => analyse(e.dataTransfer.files[0]));
    fileInput.addEventListener('change', (e) => analyse(e.target.files[0]));
    $('reportBtn').addEventListener('click', downloadReport);
    $('newCaseBtn').addEventListener('click', newCase);

    loadSamples();
    showView('intake');
});

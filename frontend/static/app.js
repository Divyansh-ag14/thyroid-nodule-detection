document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');
    const loader = document.getElementById('loader');
    const downloadBtn = document.getElementById('downloadBtn');
    const customAlert = document.getElementById('customAlert');
    const navButtons = Array.from(document.querySelectorAll('.nav-button[data-target]'));
    const reportButtons = Array.from(document.querySelectorAll('[data-action="report"]'));
    const newCaseButtons = Array.from(document.querySelectorAll('[data-action="new-case"]'));
    const reviewButtons = Array.from(document.querySelectorAll('[data-target="review"]'));
    const workspaceButtons = Array.from(document.querySelectorAll('[data-target="workspace"]'));
    const workspaceShell = document.getElementById('workspace');
    const previewPanel = document.getElementById('previewPanel');
    const reviewSection = document.getElementById('review');
    const caseBanner = document.querySelector('.case-banner');

    const selectedFileName = document.getElementById('selectedFileName');
    const selectedFileMeta = document.getElementById('selectedFileMeta');
    const analysisStatus = document.getElementById('analysisStatus');
    const analysisStatusText = document.getElementById('analysisStatusText');
    const previewState = document.getElementById('previewState');
    const scanTimestamp = document.getElementById('scanTimestamp');

    const heroLabel = document.getElementById('heroLabel');
    const clinicalSummary = document.getElementById('clinicalSummary');
    const heroConfidence = document.getElementById('heroConfidence');
    const confidenceMeter = document.getElementById('confidenceMeter');
    const confidenceLabel = document.getElementById('confidenceLabel');
    const riskBand = document.getElementById('riskBand');
    const reportSummary = document.getElementById('reportSummary');

    const predBadge = document.getElementById('predBadge');
    const confPercent = document.getElementById('confPercent');
    const clinicalInterpretation = document.getElementById('clinicalInterpretation');
    const confidenceReadout = document.getElementById('confidenceReadout');
    const reportPreviewFinding = document.getElementById('reportPreviewFinding');
    const reportPreviewInterpretation = document.getElementById('reportPreviewInterpretation');
    const opsNote = document.getElementById('opsNote');
    const analysisNote = document.getElementById('analysisNote');
    const gradcamSummary = document.getElementById('gradcamSummary');

    const originalImg = document.getElementById('originalImg');
    const gradcamImg = document.getElementById('gradcamImg');
    const heroPreview = document.getElementById('heroPreview');

    const originalStage = document.getElementById('originalStage');
    const gradcamStage = document.getElementById('gradcamStage');
    const previewStage = document.getElementById('previewStage');
    const originalPlaceholder = document.getElementById('originalPlaceholder');
    const gradcamPlaceholder = document.getElementById('gradcamPlaceholder');
    const previewEmpty = document.getElementById('previewEmpty');

    let currentFile = null;
    let currentPreviewUrl = null;
    let currentAnalysis = null;

    reportButtons.forEach((button) => {
        button.dataset.defaultLabel = button.textContent;
    });
    if (downloadBtn) {
        downloadBtn.dataset.defaultLabel = downloadBtn.textContent;
    }

    function showAlert(message, type = 'info') {
        customAlert.textContent = message;
        customAlert.className = 'alert';
        if (type === 'error') {
            customAlert.classList.add('is-error');
        } else if (type === 'success') {
            customAlert.classList.add('is-success');
        }
        customAlert.style.display = 'block';
        clearTimeout(showAlert.timeoutId);
        showAlert.timeoutId = setTimeout(() => {
            customAlert.style.display = 'none';
        }, 4200);
    }

    function setStatusTag(element, state, text) {
        element.className = 'status-tag';
        element.classList.add(state || 'neutral');
        element.textContent = text;
    }

    function setStageImage(stageElement, imgElement, placeholderElement, source) {
        if (source) {
            imgElement.src = source;
            imgElement.style.display = 'block';
            placeholderElement.style.display = 'none';
            stageElement.classList.add('has-image');
            return;
        }

        imgElement.removeAttribute('src');
        imgElement.style.display = 'none';
        placeholderElement.style.display = 'grid';
        stageElement.classList.remove('has-image');
    }

    function scrollToSection(id) {
        const element = document.getElementById(id);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function setActiveNav(id) {
        navButtons.forEach((button) => {
            button.classList.toggle('active', button.dataset.target === id);
        });
    }

    function setReportButtonsLoading(isLoading) {
        const label = isLoading ? 'Generating report...' : null;
        reportButtons.forEach((button) => {
            button.disabled = isLoading;
            button.textContent = isLoading ? label : button.dataset.defaultLabel;
        });
        if (downloadBtn) {
            downloadBtn.disabled = isLoading;
            downloadBtn.textContent = isLoading ? 'Generating report...' : downloadBtn.dataset.defaultLabel;
        }
    }

    function syncInterface() {
        const hasFile = Boolean(currentFile);
        const hasAnalysis = Boolean(currentAnalysis);
        const showWorkspace = !hasAnalysis;
        const showBanner = hasFile || hasAnalysis;

        caseBanner.classList.toggle('is-hidden', !showBanner);
        workspaceShell.classList.toggle('is-hidden', !showWorkspace);
        workspaceShell.classList.toggle('has-preview', hasFile && showWorkspace);
        dropZone.classList.toggle('has-file', hasFile);
        previewPanel.classList.toggle('is-hidden', !hasFile || !showWorkspace);
        reviewSection.classList.toggle('is-hidden', !hasAnalysis);

        reviewButtons.forEach((button) => {
            button.disabled = !hasAnalysis;
        });

        workspaceButtons.forEach((button) => {
            button.disabled = hasAnalysis;
        });

        reportButtons.forEach((button) => {
            button.disabled = !hasAnalysis;
        });

        if (downloadBtn) {
            downloadBtn.disabled = !hasAnalysis;
        }

        if (hasAnalysis) {
            setActiveNav('review');
        } else {
            setActiveNav('workspace');
        }
    }

    function formatTimestamp(date = new Date()) {
        return new Intl.DateTimeFormat('en-US', {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit'
        }).format(date);
    }

    function formatFileSize(bytes) {
        if (!Number.isFinite(bytes) || bytes <= 0) return 'Unknown size';
        const units = ['B', 'KB', 'MB', 'GB'];
        const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
        const value = bytes / 1024 ** index;
        return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
    }

    function describeConfidence(percent) {
        if (percent >= 90) return 'High-confidence output. Validate the attention map against image context before finalizing review.';
        if (percent >= 75) return 'Moderately strong output. Use the attention map and scan review together to confirm the finding.';
        if (percent >= 60) return 'Borderline confidence. Treat this as supportive evidence rather than a standalone conclusion.';
        return 'Lower-confidence output. Prioritize specialist interpretation for final assessment.';
    }

    function deriveRiskBand(result) {
        if (result.is_malignant && result.percent >= 85) return 'Elevated concern';
        if (result.is_malignant) return 'Needs close review';
        if (result.percent >= 85) return 'Lower-risk pattern';
        return 'Indeterminate-supportive';
    }

    function setLoading(isLoading) {
        loader.style.display = isLoading ? 'block' : 'none';
        dropZone.classList.toggle('active', isLoading);
    }

    function resetAnalysisState() {
        currentAnalysis = null;

        heroLabel.textContent = 'Processing image';
        clinicalSummary.textContent = 'The uploaded scan is being validated and prepared for review.';
        heroConfidence.textContent = '0.00%';
        confidenceMeter.style.width = '0%';
        confidenceLabel.textContent = 'Confidence guidance will appear after analysis is complete.';
        riskBand.textContent = 'Under analysis';
        reportSummary.textContent = 'Available after analysis';
        analysisStatusText.textContent = 'Inference running';

        predBadge.textContent = 'Awaiting analysis';
        predBadge.className = 'prediction-badge neutral';
        confPercent.textContent = '0.00%';
        clinicalInterpretation.textContent = 'Awaiting inference';
        confidenceReadout.textContent = '0.00%';
        reportPreviewFinding.textContent = 'Awaiting analysis';
        reportPreviewInterpretation.textContent = 'Awaiting inference';
        opsNote.textContent = 'Model analysis is in progress. Review output will appear here once inference completes.';
        analysisNote.textContent = 'Confidence commentary and review guidance will appear after analysis.';
        gradcamSummary.textContent = 'Bright regions indicate the image areas that most influenced the final classification.';

        setStageImage(originalStage, originalImg, originalPlaceholder, null);
        setStageImage(gradcamStage, gradcamImg, gradcamPlaceholder, null);
        syncInterface();
    }

    function startNewCase() {
        if (currentPreviewUrl) {
            URL.revokeObjectURL(currentPreviewUrl);
            currentPreviewUrl = null;
        }

        currentFile = null;
        currentAnalysis = null;
        fileInput.value = '';
        if (selectedFileName) {
            selectedFileName.textContent = 'No file selected';
        }
        if (selectedFileMeta) {
            selectedFileMeta.textContent = 'Ready for upload';
        }
        if (scanTimestamp) {
            scanTimestamp.textContent = 'Awaiting image';
        }

        heroLabel.textContent = 'No case loaded';
        clinicalSummary.textContent = 'Upload a thyroid ultrasound image to start the review workflow and populate the case summary.';
        heroConfidence.textContent = '0.00%';
        confidenceMeter.style.width = '0%';
        confidenceLabel.textContent = 'Confidence guidance will appear after analysis is complete.';
        riskBand.textContent = 'Not available';
        reportSummary.textContent = 'Awaiting analysis';
        setStatusTag(analysisStatus, 'neutral', 'Awaiting upload');
        setStatusTag(previewState, 'neutral', 'No active case');
        analysisStatusText.textContent = 'Awaiting upload';
        opsNote.textContent = 'Upload a case to populate workflow guidance.';
        predBadge.textContent = 'Awaiting analysis';
        predBadge.className = 'prediction-badge neutral';
        confPercent.textContent = '0.00%';
        clinicalInterpretation.textContent = 'Awaiting inference';
        confidenceReadout.textContent = '0.00%';
        reportPreviewFinding.textContent = 'Awaiting analysis';
        reportPreviewInterpretation.textContent = 'Awaiting inference';
        analysisNote.textContent = 'Confidence commentary and review guidance will appear after analysis.';
        gradcamSummary.textContent = 'Bright regions indicate the image areas that most influenced the final classification.';

        setStageImage(previewStage, heroPreview, previewEmpty, null);
        setStageImage(originalStage, originalImg, originalPlaceholder, null);
        setStageImage(gradcamStage, gradcamImg, gradcamPlaceholder, null);
        setLoading(false);
        syncInterface();
        scrollToSection('workspace');
    }

    function updatePreview(file) {
        if (currentPreviewUrl) {
            URL.revokeObjectURL(currentPreviewUrl);
        }
        currentPreviewUrl = URL.createObjectURL(file);
        setStageImage(previewStage, heroPreview, previewEmpty, currentPreviewUrl);
    }

    function updateFileDetails(file) {
        if (selectedFileName) {
            selectedFileName.textContent = file.name;
        }
        if (selectedFileMeta) {
            selectedFileMeta.textContent = `${formatFileSize(file.size)} ready`;
        }
        if (scanTimestamp) {
            scanTimestamp.textContent = formatTimestamp();
        }
        setStatusTag(previewState, 'live', 'Preview ready');
        setStatusTag(analysisStatus, 'live', 'Preparing case');
        analysisStatusText.textContent = 'Preview loaded';
        reportSummary.textContent = 'Pending completed analysis';
    }

    async function parseError(response, fallbackMessage) {
        try {
            const payload = await response.json();
            return payload.error || fallbackMessage;
        } catch {
            return fallbackMessage;
        }
    }

    function updateResults(result) {
        currentAnalysis = result;
        const percent = result.percent.toFixed(2);
        const risk = deriveRiskBand(result);

        heroLabel.textContent = result.is_malignant ? 'Requires close review' : 'Lower-risk pattern detected';
        clinicalSummary.textContent = result.is_malignant
            ? `The model classified this image as malignant with ${percent}% confidence. Review alongside imaging and clinical context.`
            : `The model classified this image as benign with ${percent}% confidence. Confirm the result alongside clinical context.`;
        heroConfidence.textContent = `${percent}%`;
        confidenceMeter.style.width = `${Math.max(result.percent, 4)}%`;
        confidenceLabel.textContent = describeConfidence(result.percent);
        riskBand.textContent = risk;
        reportSummary.textContent = 'Ready to download report';
        analysisStatusText.textContent = 'Analysis complete';

        predBadge.textContent = result.label;
        predBadge.className = `prediction-badge ${result.is_malignant ? 'malignant' : 'benign'}`;
        confPercent.textContent = `${percent}%`;
        clinicalInterpretation.textContent = result.is_malignant ? 'Malignant probability elevated' : 'Benign pattern detected';
        confidenceReadout.textContent = `${percent}%`;
        reportPreviewFinding.textContent = result.label;
        reportPreviewInterpretation.textContent = result.is_malignant ? 'Malignant probability elevated' : 'Benign pattern detected';
        opsNote.textContent = result.is_malignant
            ? 'Review this case closely and inspect the highlighted regions in the attention map.'
            : 'The output suggests a lower-risk pattern, but final interpretation should remain clinician-led.';
        analysisNote.textContent = describeConfidence(result.percent);

        setStatusTag(analysisStatus, 'success', 'Analysis complete');
        setStatusTag(previewState, 'success', 'Case loaded');
        if (selectedFileMeta) {
            selectedFileMeta.textContent = 'Analysis complete';
        }

        setStageImage(originalStage, originalImg, originalPlaceholder, `data:image/png;base64,${result.original_image}`);

        if (result.gradcam_image) {
            setStageImage(gradcamStage, gradcamImg, gradcamPlaceholder, `data:image/png;base64,${result.gradcam_image}`);
            gradcamSummary.textContent = 'Bright regions indicate the image areas that most influenced the final classification.';
        } else {
            setStageImage(gradcamStage, gradcamImg, gradcamPlaceholder, null);
            gradcamSummary.textContent = 'Grad-CAM visualization was not available for this run, but the classification result remains available.';
        }

        syncInterface();
    }

    async function exportReport() {
        if (!currentFile || !currentAnalysis) {
            setActiveNav('review');
            scrollToSection('review');
            showAlert('Complete an analysis before downloading the report.', 'error');
            return;
        }
        setReportButtonsLoading(true);

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/report', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(await parseError(response, 'Report generation failed.'));
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `thyroid-analysis-report-${Date.now()}.docx`;
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            URL.revokeObjectURL(url);

            reportSummary.textContent = 'Case report downloaded';
            showAlert('Case report downloaded successfully.', 'success');
        } catch (error) {
            showAlert(error.message, 'error');
        } finally {
            setReportButtonsLoading(false);
            syncInterface();
        }
    }

    async function handleFiles(files) {
        const [file] = files;
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            showAlert('Please upload a PNG or JPEG ultrasound image.', 'error');
            return;
        }

        currentFile = file;
        updateFileDetails(file);
        updatePreview(file);
        resetAnalysisState();
        setStatusTag(analysisStatus, 'live', 'Inference running');
        setLoading(true);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(await parseError(response, 'Diagnostic analysis failed. Please try again.'));
            }

            const result = await response.json();
            updateResults(result);
            scrollToSection('review');
        } catch (error) {
            if (selectedFileMeta) {
                selectedFileMeta.textContent = 'Analysis failed';
            }
            analysisStatusText.textContent = 'Analysis failed';
            heroLabel.textContent = 'Unable to analyze image';
            clinicalSummary.textContent = 'The image could not be processed. Please try another supported file.';
            setStatusTag(analysisStatus, 'error', 'Analysis failed');
            setStatusTag(previewState, 'error', 'Review needed');
            syncInterface();
            showAlert(error.message, 'error');
        } finally {
            setLoading(false);
        }
    }

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            event.stopPropagation();
        });
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('active'));
    });

    ['dragleave', 'drop'].forEach((eventName) => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('active'));
    });

    dropZone.addEventListener('drop', (event) => {
        handleFiles(event.dataTransfer.files);
    });

    fileInput.addEventListener('change', (event) => {
        handleFiles(event.target.files);
    });

    document.querySelectorAll('[data-target]').forEach((button) => {
        button.addEventListener('click', () => {
            setActiveNav(button.dataset.target);
            scrollToSection(button.dataset.target);
        });
    });

    document.querySelectorAll('[data-action="report"]').forEach((button) => {
        button.addEventListener('click', exportReport);
    });

    newCaseButtons.forEach((button) => {
        button.addEventListener('click', startNewCase);
    });

    downloadBtn.addEventListener('click', exportReport);

    const sections = ['workspace', 'review']
        .map((id) => document.getElementById(id))
        .filter(Boolean);

    if ('IntersectionObserver' in window && sections.length) {
        const observer = new IntersectionObserver((entries) => {
            const visibleEntry = entries
                .filter((entry) => entry.isIntersecting)
                .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

            if (visibleEntry?.target?.id) {
                setActiveNav(visibleEntry.target.id);
            }
        }, { threshold: 0.35 });

        sections.forEach((section) => observer.observe(section));
    }

    syncInterface();

    window.addEventListener('beforeunload', () => {
        if (currentPreviewUrl) {
            URL.revokeObjectURL(currentPreviewUrl);
        }
    });
});

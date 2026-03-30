document.addEventListener("DOMContentLoaded", () => {
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const uploadContent = document.querySelector(".upload-content");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const analyzeBtn = document.getElementById("analyze-btn");
    const removeBtn = document.getElementById("remove-btn");

    const loader = document.getElementById("loader");
    const resultsZone = document.getElementById("results-zone");
    const restartBtn = document.getElementById("restart-btn");

    let currentFile = null;

    // Trigger file select dialog
    uploadZone.addEventListener('click', (e) => {
        // Prevent click if we click analyze/remove buttons or if image is loaded
        if (e.target.tagName !== "BUTTON" && !currentFile) {
            fileInput.click();
        }
    });

    // Drag-and-drop Events
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        if(!currentFile) uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        if (!currentFile && e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadContent.style.display = "none";
            previewContainer.style.display = "flex";
            uploadZone.style.borderStyle = "solid";
            uploadZone.style.padding = "2rem";
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUI();
    });

    analyzeBtn.addEventListener('click', async (e) => {
        e.stopPropagation(); 
        if (!currentFile) return;

        // Transition to loader
        uploadZone.classList.add("hidden");
        loader.classList.remove("hidden");

        // Cycle text to indicate work is happening
        const steps = document.querySelectorAll('.step');
        let currentStep = 0;
        const interval = setInterval(() => {
            if(currentStep < steps.length - 1) {
                steps[currentStep].classList.remove('active');
                currentStep++;
                steps[currentStep].classList.add('active');
            }
        }, 1500);

        try {
            const formData = new FormData();
            formData.append('file', currentFile);
            
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            clearInterval(interval);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Server format error");
            }
            if(!data.success) {
                throw new Error(data.error || "Analysis pipeline failed to start.");
            }

            renderResults(data);

        } catch (error) {
            clearInterval(interval);
            alert("Error: " + error.message);
            resetUI();
        }
    });

    function renderResults(data) {
        loader.classList.add("hidden");
        resultsZone.classList.remove("hidden");

        const finalVerdictBanner = document.getElementById("final-verdict-banner");
        const finalVerdictText = document.getElementById("final-verdict");
        const finalScoreText = document.getElementById("final-score");
        const facesDetected = document.getElementById("faces-detected");
        const toolsGrid = document.getElementById("tools-grid");

        toolsGrid.innerHTML = '';

        const scorePercent = (data.final_score * 100).toFixed(1);
        finalScoreText.textContent = `${scorePercent}%`;
        facesDetected.textContent = data.faces_detected;
        
        finalVerdictBanner.classList.remove("fake", "real");
        if (data.is_fake) {
            finalVerdictText.textContent = "⚠️ FAKE MEDIA DETECTED";
            finalVerdictBanner.classList.add("fake");
        } else {
            finalVerdictText.textContent = "✅ AUTHENTIC MEDIA";
            finalVerdictBanner.classList.add("real");
        }

        const renderToolName = (name) => {
            return name.replace('run_', '').replace('check_', '').replace('_', ' ');
        };

        // Render tool cards logically
        data.results.forEach(res => {
            let statusClass = "status-error";
            let statusText = "ERROR";

            if (res.success) {
                const isRisk = res.score > 0.5;
                statusClass = isRisk ? "status-invalid" : "status-valid";
                statusText = isRisk ? "SUSPICIOUS" : "CLEAR";
            }

            const card = document.createElement('div');
            card.className = "tool-card";
            
            let cardInner = `
                <div class="tool-header">
                    <div class="tool-name">${renderToolName(res.tool_name)}</div>
                    <div class="tool-status ${statusClass}">${statusText}</div>
                </div>`;
            
            if (res.success) {
                cardInner += `
                <div class="tool-metrics">
                    <div class="metric">
                        <span class="metric-label">Score</span>
                        <span class="metric-value" style="color: ${res.score > 0.5 ? 'var(--alert)' : 'var(--success)'}">${(res.score).toFixed(2)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Confidence</span>
                        <span class="metric-value">${(res.confidence).toFixed(2)}</span>
                    </div>
                </div>`;
            }

            cardInner += `
                <div class="tool-evidence">
                    ${res.success ? res.evidence_summary : (res.error_msg || "Tool execution failed.")}
                </div>
            `;
            
            card.innerHTML = cardInner;
            toolsGrid.appendChild(card);
        });
    }

    function resetUI() {
        currentFile = null;
        fileInput.value = "";
        uploadContent.style.display = "block";
        previewContainer.style.display = "none";
        
        uploadZone.classList.remove("hidden");
        loader.classList.add("hidden");
        resultsZone.classList.add("hidden");
        
        uploadZone.style.borderStyle = "dashed";
        uploadZone.style.padding = "3rem 2rem";
        
        // Reset active steps
        const steps = document.querySelectorAll('.step');
        steps.forEach(s => s.classList.remove('active'));
        steps[0].classList.add('active');
    }

    restartBtn.addEventListener('click', resetUI);
});

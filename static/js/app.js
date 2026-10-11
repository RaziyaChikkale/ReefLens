/**
 * ReefLens — Marine Species AI Vision Platform
 * Client Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // =========================================================
  // APPLICATION STATE
  // =========================================================
  const state = {
    selectedFile: null,
    selectedSample: null, // { species, filename, url }
    isVideoPaused: false,
    currentVideoEngine: "knn",
    metricsData: null,
    sampleList: []
  };

  // DOM Elements
  const tabs = document.querySelectorAll(".nav-tab");
  const tabContents = document.querySelectorAll(".tab-content");

  const dropZone = document.getElementById("image-drop-zone");
  const fileInput = document.getElementById("image-file-input");
  const dropPrompt = document.getElementById("drop-prompt");
  const dropPreviewContainer = document.getElementById("drop-preview-container");
  const imagePreview = document.getElementById("image-preview");
  const btnClearImage = document.getElementById("btn-clear-image");
  const btnRunInference = document.getElementById("btn-run-inference");

  const samplesGalleryList = document.getElementById("samples-gallery-list");
  const btnReloadSamples = document.getElementById("reload-samples-btn");

  const resultsEmptyState = document.getElementById("results-empty-state");
  const resultsLoadingState = document.getElementById("results-loading-state");
  const resultsPopulated = document.getElementById("results-populated");
  const inferenceTimeBadge = document.getElementById("inference-time-badge");
  const inferenceTimeText = document.getElementById("inference-time-text");

  const resultSpeciesName = document.getElementById("result-species-name");
  const resultScientificName = document.getElementById("result-scientific-name");
  const resultConfidenceVal = document.getElementById("result-confidence-val");
  const resultAnnotatedImg = document.getElementById("result-annotated-img");
  const resultModelViewImg = document.getElementById("result-model-view-img");
  const rankingsBarsContainer = document.getElementById("rankings-bars-container");

  const resultSpeciesDesc = document.getElementById("result-species-desc");
  const resultHabitat = document.getElementById("result-habitat");
  const resultDepth = document.getElementById("result-depth");
  const resultDiet = document.getElementById("result-diet");
  const resultStatus = document.getElementById("result-status");

  // Video elements
  const liveVideoStream = document.getElementById("live-video-stream");
  const engineSelect = document.getElementById("engine-select");
  const btnToggleVideo = document.getElementById("btn-toggle-video");
  const btnToggleVideoText = document.getElementById("btn-toggle-video-text");
  const btnRestartVideo = document.getElementById("btn-restart-video");
  const hudEngineName = document.getElementById("hud-engine-name");

  // Metrics elements
  const metricsTableBody = document.getElementById("metrics-table-body");

  // =========================================================
  // 1. TAB NAVIGATION
  // =========================================================
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.dataset.tab;

      tabs.forEach(t => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");

      tabContents.forEach(content => {
        if (content.id === `section-${targetId}`) {
          content.classList.remove("hidden");
          content.classList.add("active");
        } else {
          content.classList.add("hidden");
          content.classList.remove("active");
        }
      });

      // Lazy load metrics when switching to metrics tab
      if (targetId === "metrics" && !state.metricsData) {
        loadMetrics();
      }
    });
  });

  // =========================================================
  // 2. QUICK TEST GALLERY (Real Dataset Samples)
  // =========================================================
  async function loadSamples() {
    try {
      samplesGalleryList.innerHTML = `
        <div class="loading-spinner-row">
          <div class="spinner"></div>
          <span>Loading verified dataset specimens...</span>
        </div>
      `;

      const response = await fetch("/api/samples");
      const data = await response.json();
      state.sampleList = data.samples || [];

      if (state.sampleList.length === 0) {
        samplesGalleryList.innerHTML = `<div class="loading-spinner-row">No sample images found in dataset.</div>`;
        return;
      }

      samplesGalleryList.innerHTML = "";
      state.sampleList.forEach(sample => {
        const chip = document.createElement("div");
        chip.className = "sample-chip";
        chip.title = `Classify this ${sample.species} specimen`;
        chip.innerHTML = `
          <img src="${sample.url}" alt="${sample.species}" class="sample-thumbnail" loading="lazy">
          <span class="sample-label">${sample.species}</span>
          <span class="sample-count">${sample.total_images} photos</span>
        `;

        chip.addEventListener("click", () => {
          selectSample(sample, chip);
        });

        samplesGalleryList.appendChild(chip);
      });
    } catch (err) {
      console.error("Error loading samples:", err);
      samplesGalleryList.innerHTML = `<div class="loading-spinner-row">Error connecting to dataset.</div>`;
    }
  }

  function selectSample(sample, chipElement) {
    // Highlight chip
    document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
    if (chipElement) chipElement.classList.add("active");

    state.selectedSample = sample;
    state.selectedFile = null;

    // Show preview in drop zone
    imagePreview.src = sample.url;
    dropPrompt.classList.add("hidden");
    dropPreviewContainer.classList.remove("hidden");
    btnRunInference.disabled = false;

    // Immediately trigger classification for instant test
    runInference();
  }

  btnReloadSamples.addEventListener("click", loadSamples);

  // =========================================================
  // 3. IMAGE UPLOAD & DRAG-AND-DROP
  // =========================================================
  dropZone.addEventListener("click", (e) => {
    // Don't trigger if clicked on clear button
    if (e.target.closest("#btn-clear-image")) return;
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelection(file);
  });

  // Drag and Drop
  ["dragenter", "dragover"].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("drag-over");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleFileSelection(file);
    }
  });

  // Paste image from clipboard
  window.addEventListener("paste", (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let item of items) {
      if (item.type.indexOf("image") !== -1) {
        const file = item.getAsFile();
        handleFileSelection(file);
        break;
      }
    }
  });

  function handleFileSelection(file) {
    state.selectedFile = file;
    state.selectedSample = null;
    document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropPrompt.classList.add("hidden");
      dropPreviewContainer.classList.remove("hidden");
      btnRunInference.disabled = false;
      // Run inference automatically
      runInference();
    };
    reader.readAsDataURL(file);
  }

  btnClearImage.addEventListener("click", (e) => {
    e.stopPropagation();
    state.selectedFile = null;
    state.selectedSample = null;
    fileInput.value = "";
    imagePreview.src = "";
    dropPrompt.classList.remove("hidden");
    dropPreviewContainer.classList.add("hidden");
    btnRunInference.disabled = true;

    // Reset results view
    resultsEmptyState.classList.remove("hidden");
    resultsLoadingState.classList.add("hidden");
    resultsPopulated.classList.add("hidden");
    inferenceTimeBadge.classList.add("hidden");
    document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
  });

  btnRunInference.addEventListener("click", () => {
    runInference();
  });

  // =========================================================
  // 4. CLASSIFICATION INFERENCE CALL
  // =========================================================
  async function runInference() {
    if (!state.selectedFile && !state.selectedSample) return;

    // Update UI states
    resultsEmptyState.classList.add("hidden");
    resultsPopulated.classList.add("hidden");
    resultsLoadingState.classList.remove("hidden");
    btnRunInference.disabled = true;

    const t0 = performance.now();

    try {
      let response;

      if (state.selectedSample) {
        // Send sample species & filename
        const formData = new FormData();
        formData.append("sample_species", state.selectedSample.species);
        formData.append("sample_filename", state.selectedSample.filename);
        response = await fetch("/api/classify", {
          method: "POST",
          body: formData
        });
      } else if (state.selectedFile) {
        // Send uploaded file
        const formData = new FormData();
        formData.append("file", state.selectedFile);
        response = await fetch("/api/classify", {
          method: "POST",
          body: formData
        });
      }

      const resJson = await response.json();
      const elapsed = Math.round(performance.now() - t0);

      if (resJson.success) {
        renderInferenceResults(resJson.data, elapsed);
      } else {
        alert("Classification Error: " + (resJson.error || "Unknown error"));
        resultsLoadingState.classList.add("hidden");
        resultsEmptyState.classList.remove("hidden");
      }
    } catch (err) {
      console.error("Error during inference:", err);
      alert("Inference connection failed. Check server status.");
      resultsLoadingState.classList.add("hidden");
      resultsEmptyState.classList.remove("hidden");
    } finally {
      btnRunInference.disabled = false;
    }
  }

  function renderInferenceResults(data, elapsedMs) {
    resultsLoadingState.classList.add("hidden");
    resultsPopulated.classList.remove("hidden");

    // Latency badge
    inferenceTimeText.textContent = `${elapsedMs} ms latency`;
    inferenceTimeBadge.classList.remove("hidden");

    // Hero title & confidence
    resultSpeciesName.textContent = data.species;
    resultScientificName.textContent = data.species_info.scientific_name || "Marine Specimen";
    resultConfidenceVal.textContent = `${data.confidence}%`;

    // Multi-views
    resultAnnotatedImg.src = data.annotated_image;
    resultModelViewImg.src = data.model_input_image;

    // Top rankings probability bars
    rankingsBarsContainer.innerHTML = "";
    data.rankings.forEach((rank, idx) => {
      const isTop = idx === 0;
      const rankItem = document.createElement("div");
      rankItem.className = "rank-item";
      rankItem.innerHTML = `
        <div class="rank-header">
          <span class="rank-species">${isTop ? "★ " : ""}${rank.species}</span>
          <span class="rank-pct">${rank.confidence}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width: 0%; background: ${isTop ? "linear-gradient(90deg, #0077b6, #00f5d4)" : "rgba(255,255,255,0.2)"};"></div>
        </div>
      `;
      rankingsBarsContainer.appendChild(rankItem);

      // Animate width bar
      setTimeout(() => {
        const fill = rankItem.querySelector(".bar-fill");
        if (fill) fill.style.width = `${Math.max(rank.confidence, 3)}%`;
      }, 50 + idx * 40);
    });

    // Marine Facts Profile
    const info = data.species_info || {};
    resultSpeciesDesc.textContent = info.description || "Ocean fauna specimen captured in marine habitat.";
    resultHabitat.textContent = info.habitat || "Marine Ecosystem";
    resultDepth.textContent = info.depth || "Coastal to Pelagic";
    resultDiet.textContent = info.diet || "Invertebrates / Plankton";
    resultStatus.textContent = info.status || "Protected Marine Life";
  }

  // =========================================================
  // 5. LIVE VIDEO STREAM CONTROLS
  // =========================================================
  function updateVideoStream() {
    if (state.isVideoPaused) {
      liveVideoStream.src = "";
    } else {
      liveVideoStream.src = `/api/video-stream?engine=${state.currentVideoEngine}&t=${Date.now()}`;
    }
  }

  engineSelect.addEventListener("change", (e) => {
    state.currentVideoEngine = e.target.value;
    hudEngineName.textContent = `Engine: ${state.currentVideoEngine.toUpperCase()}`;
    updateVideoStream();
  });

  btnToggleVideo.addEventListener("click", () => {
    state.isVideoPaused = !state.isVideoPaused;
    if (state.isVideoPaused) {
      btnToggleVideoText.textContent = "Resume Stream";
      liveVideoStream.src = "";
    } else {
      btnToggleVideoText.textContent = "Pause Stream";
      updateVideoStream();
    }
  });

  btnRestartVideo.addEventListener("click", () => {
    state.isVideoPaused = false;
    btnToggleVideoText.textContent = "Pause Stream";
    updateVideoStream();
  });

  // =========================================================
  // 6. METRICS & DATASET ANALYTICS
  // =========================================================
  async function loadMetrics() {
    try {
      const response = await fetch("/api/metrics");
      const data = await response.json();
      state.metricsData = data;

      metricsTableBody.innerHTML = "";
      const classesData = data.classes_breakdown || {};

      for (const [speciesName, metrics] of Object.entries(classesData)) {
        const row = document.createElement("tr");
        const f1 = metrics.f1;
        const qualityBadge = f1 >= 0.90
          ? `<span class="badge-excellent">Optimal (90%+)</span>`
          : `<span class="badge-good">High (${Math.round(f1 * 100)}%)</span>`;

        row.innerHTML = `
          <td><span class="species-cell">${speciesName}</span></td>
          <td class="metric-score">${metrics.dataset_count} imgs</td>
          <td class="metric-score">${(metrics.precision * 100).toFixed(0)}%</td>
          <td class="metric-score">${(metrics.recall * 100).toFixed(0)}%</td>
          <td class="metric-score" style="color: var(--accent-cyan);">${(metrics.f1 * 100).toFixed(0)}%</td>
          <td class="metric-score">${metrics.support}</td>
          <td>${qualityBadge}</td>
        `;
        metricsTableBody.appendChild(row);
      }
    } catch (err) {
      console.error("Error loading metrics:", err);
    }
  }

  // =========================================================
  // 7. CLI COPY-TO-CLIPBOARD HELPER
  // =========================================================
  document.querySelectorAll(".btn-copy").forEach(btn => {
    btn.addEventListener("click", () => {
      const textToCopy = btn.dataset.copy;
      if (textToCopy) {
        navigator.clipboard.writeText(textToCopy).then(() => {
          const originalText = btn.textContent;
          btn.textContent = "Copied!";
          btn.style.background = "var(--accent-cyan)";
          btn.style.color = "#000";
          setTimeout(() => {
            btn.textContent = originalText;
            btn.style.background = "";
            btn.style.color = "";
          }, 1800);
        });
      }
    });
  });

  // =========================================================
  // INITIALIZATION
  // =========================================================
  loadSamples();
});

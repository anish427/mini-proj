/* ================================================================
   DeepFake Tracker — Frontend Logic
   ================================================================ */
// if (!localStorage.getItem('dft_current')) window.location.href = 'auth.html';
const API_BASE = "";
const $ = (id) => document.getElementById(id);

const dropzone = $("dropzone");
const fileInput = $("file-input");
const analyzeBtn = $("analyze-btn");
const previewPlaceholder = $("preview-placeholder");
const previewImg = $("preview-img");
const previewVideo = $("preview-video");
const resultArea = $("result-area");
const overlay = $("loading-overlay");
const forensicsSection = $("forensics-section");
const forensicsGrid = $("forensics-grid");

let selectedFile = null;
let objectUrl = null;

/* --- Helpers --- */

function setLoading(on) {
  overlay.classList.toggle("visible", on);
  overlay.setAttribute("aria-hidden", on ? "false" : "true");
  analyzeBtn.disabled = on || !selectedFile;
}

function clearPreviewMedia() {
  previewImg.hidden = true;
  previewVideo.hidden = true;
  previewVideo.pause();
  previewVideo.removeAttribute("src");
  previewVideo.load();
  previewImg.removeAttribute("src");
  if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
}

function showPlaceholder(text) {
  clearPreviewMedia();
  previewPlaceholder.hidden = false;
  previewPlaceholder.querySelector("span").textContent = text;
}

function isVideo(f) { return f.type.startsWith("video/"); }
function isImage(f) { return f.type.startsWith("image/"); }

function onFile(file) {
  if (!file || (!isImage(file) && !isVideo(file))) {
    showPlaceholder("Choose an image or video file.");
    selectedFile = null;
    analyzeBtn.disabled = true;
    return;
  }
  selectedFile = file;
  analyzeBtn.disabled = false;
  previewPlaceholder.hidden = true;
  objectUrl = URL.createObjectURL(file);
  if (isImage(file)) {
    previewVideo.hidden = true; previewVideo.pause();
    previewImg.hidden = false; previewImg.src = objectUrl;
  } else {
    previewImg.hidden = true; previewImg.removeAttribute("src");
    previewVideo.hidden = false; previewVideo.src = objectUrl;
  }
}

function pctStr(p) {
  if (typeof p !== "number" || Number.isNaN(p)) return "0.00";
  return Math.min(100, Math.max(0, p * 100)).toFixed(2);
}

function escapeHtml(s) {
  const d = document.createElement("div"); d.textContent = s; return d.innerHTML;
}

function severity(score) {
  if (score < 0.35) return "low";
  if (score < 0.60) return "mid";
  return "high";
}

/* --- Render Result --- */

function renderResult(data, isVideoResult) {
  const label = data.label || "Unknown";
  const pr = typeof data.prob_real === "number" ? data.prob_real : null;
  const pf = typeof data.prob_fake === "number" ? data.prob_fake : null;
  const conf = typeof data.confidence === "number" ? data.confidence
    : pr != null && pf != null ? (label === "Fake" ? pf : pr) : 0;

  const isFake = label === "Fake" || label === "Deepfake" || label === "AI-Edited";
  const cls = isFake ? "fake" : "real";
  const icon = isFake ? "⚠" : "✓";
  const confPct = pctStr(conf);
  const realPct = pr != null ? pctStr(pr) : "0.00";
  const fakePct = pf != null ? pctStr(pf) : "0.00";
  const gaugeColor = isFake ? "var(--fake)" : "var(--real)";
  const gaugeAngle = Math.round(conf * 360);

  let videoMeta = "";
  if (isVideoResult) {
    videoMeta += `<div class="video-meta">`;
    if (data.frames_analyzed != null)
      videoMeta += `<p class="meta-line">Frames: <code>${data.frames_analyzed}</code></p>`;
    if (data.frame_prob_fake_std != null)
      videoMeta += `<p class="meta-line">Frame σ: <code>${Number(data.frame_prob_fake_std).toFixed(4)}</code></p>`;
    if (data.face_detection_ratio != null)
      videoMeta += `<p class="meta-line">Faces: <code>${pctStr(data.face_detection_ratio)}%</code></p>`;
    videoMeta += `</div>`;
  }

  resultArea.innerHTML = `
    <div class="animate-in">
      <div class="verdict">
        <div class="verdict-icon ${cls}">${icon}</div>
        <div>
          <div class="verdict-label ${cls}">${escapeHtml(label)}</div>
          <div class="verdict-conf">${confPct}% confidence</div>
        </div>
      </div>
      <div class="gauge-container">
        <div class="gauge">
          <div class="gauge-ring" style="background: conic-gradient(${gaugeColor} ${gaugeAngle}deg, rgba(0,0,0,0.3) ${gaugeAngle}deg)"></div>
          <div class="gauge-inner">
            <div class="gauge-value ${cls}">${confPct}%</div>
            <div class="gauge-label">${isFake ? "FAKE" : "REAL"}</div>
          </div>
        </div>
      </div>
      <div class="prob-bars">
        <div class="prob-row">
          <div class="prob-row-header"><span>Real</span><span>${realPct}%</span></div>
          <div class="prob-bar real"><span style="width:${realPct}%"></span></div>
        </div>
        <div class="prob-row">
          <div class="prob-row-header"><span>Fake</span><span>${fakePct}%</span></div>
          <div class="prob-bar fake"><span style="width:${fakePct}%"></span></div>
        </div>
      </div>
      <p class="meta-line">Engine: <code>${escapeHtml(data.backend || "unknown")}</code></p>
      ${videoMeta}
    </div>`;

  if (data.signals) renderForensics(data.signals);
  else forensicsSection.hidden = true;
}

/* --- Generic Signal Card Renderer --- */

function renderSignalCard(sig) {
  // Determine if this is a neural signal (has prob_fake) or forensic (has score)
  const isNeural = sig.prob_fake !== undefined;
  const fakeVal = isNeural ? sig.prob_fake : sig.score;
  const sev = severity(fakeVal);

  let details = "";

  if (isNeural) {
    details = `Fake: <code>${pctStr(sig.prob_fake)}%</code> · Real: <code>${pctStr(sig.prob_real)}%</code>`;
  } else if (sig.mean !== undefined) {
    // ELA
    details = `Mean: <code>${sig.mean.toFixed(2)}</code> · Std: <code>${sig.std.toFixed(2)}</code><br/>
      ${fakeVal < 0.35 ? "Compression looks natural" : fakeVal < 0.6 ? "Some anomalies" : "Significant anomalies"}`;
  } else if (sig.findings) {
    // Metadata
    details = `<ul class="findings-list">${sig.findings.map(f => `<li>${escapeHtml(f)}</li>`).join("")}</ul>`;
  } else if (sig.hf_ratio !== undefined) {
    // Frequency
    details = `HF ratio: <code>${sig.hf_ratio.toFixed(4)}</code><br/>
      ${fakeVal < 0.35 ? "Natural camera noise" : fakeVal < 0.6 ? "Moderate pattern" : "Low HF energy — AI typical"}`;
  } else if (sig.laplacian_var !== undefined) {
    // Texture
    details = `Laplacian: <code>${sig.laplacian_var.toFixed(1)}</code> · Smooth: <code>${pctStr(sig.smooth_ratio)}%</code><br/>
      ${fakeVal < 0.35 ? "Natural texture detail" : fakeVal < 0.6 ? "Somewhat smooth" : "Unnaturally smooth — GAN typical"}`;
  }

  return `
    <div class="signal-card">
      <div class="signal-header">
        <div class="signal-name"><span class="signal-dot ${sev}"></span>${escapeHtml(sig.name)}</div>
        <div class="signal-score ${sev}">${pctStr(fakeVal)}%</div>
      </div>
      <div class="signal-bar"><span class="${sev}" style="width:${pctStr(fakeVal)}%"></span></div>
      <div class="signal-detail">${details}</div>
    </div>`;
}

function renderForensics(signals) {
  forensicsSection.hidden = false;
  forensicsSection.classList.add("animate-in");

  // Render in order: neural models first, then forensic
  const order = ["siglip", "vit", "sdxl", "ela", "metadata", "frequency", "texture"];
  let html = "";
  for (const key of order) {
    if (signals[key]) html += renderSignalCard(signals[key]);
  }
  // Any remaining signals not in the order
  for (const [key, sig] of Object.entries(signals)) {
    if (!order.includes(key)) html += renderSignalCard(sig);
  }

  forensicsGrid.innerHTML = html;
}

/* --- Error --- */
function renderError(msg) {
  resultArea.innerHTML = `<div class="error-msg animate-in">${escapeHtml(msg)}</div>`;
  forensicsSection.hidden = true;
}

/* --- Analyze --- */
async function analyze() {
  if (!selectedFile) return;
  const form = new FormData();
  form.append("file", selectedFile);
  const video = isVideo(selectedFile);
  const url = video ? `${API_BASE}/api/detect-video` : `${API_BASE}/api/detect-image`;

  setLoading(true);
  resultArea.innerHTML = '<div class="result-empty"><span>Analyzing…</span></div>';
  forensicsSection.hidden = true;

  try {
    const res = await fetch(url, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok || data.ok === false) {
      renderError(data.error || `Request failed (${res.status})`);
      return;
    }
    renderResult(data, video);
  } catch (e) {
    renderError(e.message || "Network error. Is the Flask server running?");
  } finally {
    setLoading(false);
  }
}

/* --- Events --- */
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", e => {
  e.preventDefault(); dropzone.classList.remove("dragover");
  const f = e.dataTransfer.files[0];
  if (f) { fileInput.files = e.dataTransfer.files; onFile(f); }
});
fileInput.addEventListener("change", () => onFile(fileInput.files[0]));
analyzeBtn.addEventListener("click", e => { e.preventDefault(); e.stopPropagation(); analyze(); });

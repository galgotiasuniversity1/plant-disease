const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
let blob = null, stream = null, cfg = { has_key: true }, last = null;

/* ---------- init ---------- */
fetch("/api/config").then((r) => r.json()).then((c) => {
  cfg = c;
  $("#owner").textContent = $("#owner2").textContent = c.owner;
  $("#apiKey").value = localStorage.getItem("groq_key") || "";
}).catch(() => {});

function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast.id); toast.id = setTimeout(() => t.classList.add("hidden"), 4500);
}

/* ---------- input tabs ---------- */
document.querySelectorAll("[data-tab]").forEach((b) => b.onclick = () => {
  document.querySelectorAll("[data-tab]").forEach((x) => x.classList.toggle("active", x === b));
  const cam = b.dataset.tab === "camera";
  $("#pane-browse").hidden = cam; $("#pane-camera").hidden = !cam;
  cam ? startCam() : stopCam();
});

$("#file").onchange = (e) => e.target.files[0] && setImage(e.target.files[0]);
$("#fileCam").onchange = (e) => e.target.files[0] && setImage(e.target.files[0]);
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; f && setImage(f); });

/* ---------- camera ---------- */
async function startCam() {
  const msg = $("#camMsg"); msg.classList.remove("off"); msg.textContent = "Starting camera…";
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false });
    $("#video").srcObject = stream; msg.classList.add("off");
  } catch (e) {
    msg.textContent = "Camera unavailable (" + (e.name || "blocked") + "). Allow camera access, or use the phone camera button below.";
  }
}
function stopCam() { stream && stream.getTracks().forEach((t) => t.stop()); stream = null; }
$("#snap").onclick = () => {
  const v = $("#video");
  if (!stream || !v.videoWidth) return toast("Camera isn't ready yet.");
  const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  c.toBlob((b) => setImage(b), "image/jpeg", 0.92);
};

/* ---------- check + preview ---------- */
async function setImage(b) {
  blob = b; stopCam();
  $("#inputCard").classList.add("hidden"); $("#results").classList.add("hidden");
  $("#previewCard").classList.remove("hidden"); $("#analyze").hidden = true; $("#keyRow").hidden = true;
  $("#preview").src = URL.createObjectURL(b);
  $("#checks").innerHTML = '<div class="status"><span class="spin"></span> Checking image…</div>';
  const fd = new FormData(); fd.append("image", b, "leaf.jpg");
  try {
    const r = await (await fetch("/api/check", { method: "POST", body: fd })).json();
    renderChecks(r);
  } catch { $("#checks").innerHTML = '<div class="status bad">Could not reach the server.</div>'; }
}

function metricsHtml(m) {
  if (!m) return "";
  return "<details><summary>Measurements</summary>" + Object.entries(m).map(([k, v]) => `${esc(k)}: ${esc(v)}`).join(" · ") + "</details>";
}
function renderChecks(r) {
  const box = $("#checks");
  if (r.ok) {
    box.innerHTML = `<div class="status ok">✅ ${esc(r.message)}${metricsHtml(r.metrics)}</div>`;
    $("#analyze").hidden = false; $("#keyRow").hidden = cfg.has_key;
    return;
  }
  const icon = r.stage === "quality" ? "⚠️" : "❌";
  const title = r.stage === "leaf" ? "Not a leaf image. " : "";
  const issues = r.issues ? "<ul>" + r.issues.map((i) => `<li>${esc(i)}</li>`).join("") + "</ul>" : "";
  box.innerHTML = `<div class="status ${r.stage === "quality" ? "warn" : "bad"}">${icon} <b>${title}</b>${esc(r.message)}${issues}${metricsHtml(r.metrics)}<small>Nothing was sent to the AI. Retake the photo and try again.</small></div>`;
}

$("#reset").onclick = () => {
  blob = null; $("#previewCard").classList.add("hidden"); $("#results").classList.add("hidden");
  $("#inputCard").classList.remove("hidden"); $("#file").value = ""; $("#fileCam").value = "";
  if (!$("#pane-camera").hidden) startCam();
};

/* ---------- analyze ---------- */
$("#analyze").onclick = async () => {
  const btn = $("#analyze"), key = $("#apiKey").value.trim();
  if (!cfg.has_key && !key) return toast("Enter your Groq API key first.");
  if (key) localStorage.setItem("groq_key", key);
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span> Analyzing…';
  const fd = new FormData(); fd.append("image", blob, "leaf.jpg");
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd, headers: key ? { "X-API-Key": key } : {} });
    const d = await res.json();
    if (!res.ok || !d.ok) {
      if (d.stage === "leaf" || d.stage === "quality") renderChecks(d); else toast(d.message || "Analysis failed.");
      return;
    }
    last = d; renderResults(d);
  } catch { toast("Could not reach the server."); }
  finally { btn.disabled = false; btn.innerHTML = "🔍 Analyze leaf"; }
};

/* ---------- results ---------- */
function list(a) { return "<ul>" + (a || []).map((x) => `<li>${esc(x)}</li>`).join("") + "</ul>"; }
function renderResults(d) {
  const r = d.result, conf = Number(r.confidence) || 0;
  $("#mPlant").textContent = r.plant || "-"; $("#mCond").textContent = r.condition || "-";
  $("#mConf").textContent = conf.toFixed(1) + "%"; $("#mSev").textContent = r.severity || "-";
  $("#lowConf").classList.toggle("hidden", conf >= 40);

  const tabs = { overview: `<p>${esc(r.overview)}</p>`, symptoms: list(r.symptoms), treatment: list(r.treatment), prevention: list(r.prevention), explanation: `<p>${esc(r.explanation)}</p>` };
  const show = (k) => {
    document.querySelectorAll("[data-info]").forEach((b) => b.classList.toggle("active", b.dataset.info === k));
    $("#infoBody").innerHTML = tabs[k];
  };
  document.querySelectorAll("[data-info]").forEach((b) => b.onclick = () => show(b.dataset.info));
  show("overview");

  $("#preds").innerHTML = (r.top_predictions || []).map((p) => {
    const c = Math.max(0, Math.min(100, Number(p.confidence) || 0));
    return `<div class="pred"><div class="l"><b>${esc(p.label)}</b><span>${c.toFixed(1)}%</span></div><div class="bar"><i style="width:0" data-w="${c}"></i></div></div>`;
  }).join("");
  requestAnimationFrame(() => document.querySelectorAll(".bar i").forEach((i) => i.style.width = i.dataset.w + "%"));

  $("#imgOrig").src = d.original; $("#imgHeat").src = d.heatmap;
  $("#heatCap").textContent = `Affected-area map (~${Math.round(d.affected_pct)}% of leaf)`;
  $("#results").classList.remove("hidden");
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- PDF ---------- */
$("#pdf").onclick = async () => {
  if (!last) return;
  const btn = $("#pdf"); btn.disabled = true;
  try {
    const res = await fetch("/api/report", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ result: last.result, original: last.original, heatmap: last.heatmap }) });
    if (!res.ok) throw new Error((await res.json()).message);
    const a = document.createElement("a"); a.href = URL.createObjectURL(await res.blob());
    a.download = "leaf_disease_report.pdf"; a.click(); URL.revokeObjectURL(a.href);
  } catch (e) { toast(e.message || "PDF failed."); }
  finally { btn.disabled = false; }
};

/* ================= UI polish (visual only; no logic changes above) ================= */
// floating leaves
(() => {
  const box = document.getElementById("leaves"), icons = ["🍃", "🌿", "🍀", "🍂", "🌱"];
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  for (let i = 0; i < 16; i++) {
    const s = document.createElement("span");
    s.textContent = icons[i % icons.length];
    s.style.cssText = `left:${Math.random() * 100}%;font-size:${14 + Math.random() * 20}px;animation-duration:${14 + Math.random() * 16}s;animation-delay:-${Math.random() * 25}s`;
    box.appendChild(s);
  }
})();

// scanning beam while the AI is working
(() => {
  const btn = document.getElementById("analyze"), box = document.querySelector(".scanbox");
  btn.addEventListener("click", () => box.classList.add("scanning"));
  new MutationObserver(() => { if (!btn.disabled) box.classList.remove("scanning"); })
    .observe(btn, { attributes: true, attributeFilter: ["disabled"] });
})();

// confidence ring + count-up after results render
const _renderResults = renderResults;
renderResults = function (d) {
  _renderResults(d);
  const target = Math.max(0, Math.min(100, Number(d.result.confidence) || 0));
  const ring = document.getElementById("ring"), label = document.getElementById("mConf");
  ring.style.strokeDashoffset = 326.7;
  requestAnimationFrame(() => requestAnimationFrame(() => { ring.style.strokeDashoffset = 326.7 * (1 - target / 100); }));
  const t0 = performance.now();
  (function tick(t) {
    const p = Math.min(1, (t - t0) / 1200);
    label.textContent = (target * (1 - Math.pow(1 - p, 3))).toFixed(1) + "%";
    if (p < 1) requestAnimationFrame(tick);
  })(t0);
};

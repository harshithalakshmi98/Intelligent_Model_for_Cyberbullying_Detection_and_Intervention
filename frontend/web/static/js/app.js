const form = document.getElementById("predict-form");
const resultEl = document.getElementById("result");
const queueList = document.getElementById("queue-list");
const queueEmpty = document.getElementById("queue-empty");
const healthBadge = document.getElementById("health-badge");
const useRelational = document.getElementById("use-relational");
const relationalFields = document.getElementById("relational-fields");

useRelational.addEventListener("change", () => {
  relationalFields.style.opacity = useRelational.checked ? "1" : "0.45";
  relationalFields.querySelectorAll("input").forEach((el) => {
    el.disabled = !useRelational.checked;
  });
});

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.models_loaded) {
      healthBadge.textContent = "Models ready";
      healthBadge.className = "badge badge-ok";
    } else {
      healthBadge.textContent = "Models missing";
      healthBadge.className = "badge badge-error";
    }
  } catch {
    healthBadge.textContent = "API offline";
    healthBadge.className = "badge badge-error";
  }
}

function tierClass(tier) {
  if (tier === 1) return "tier-1";
  if (tier === 2) return "tier-2";
  if (tier === 3) return "tier-3";
  return "";
}

function renderResult(data) {
  resultEl.classList.remove("hidden");
  const pct = (data.probability * 100).toFixed(1);
  resultEl.innerHTML = `
    <div class="tier ${tierClass(data.tier)}">${data.tier_label}</div>
    <p><strong>Model:</strong> ${data.model_id} &nbsp;|&nbsp;
       <strong>Prediction:</strong> ${data.predicted_cyberbullying ? "Cyberbullying" : "Not cyberbullying"}
       &nbsp;|&nbsp; <strong>Score:</strong> ${pct}%</p>
    <p class="muted">${data.tier_reason}</p>
    ${data.combined_score != null ? `<p><strong>combined_score:</strong> ${data.combined_score.toFixed(3)}
       &nbsp;|&nbsp; <strong>peerness:</strong> ${data.peerness?.toFixed(3) ?? "—"}
       &nbsp;|&nbsp; <strong>repeat offender:</strong> ${data.is_repeat_offender ? "yes" : "no"}</p>` : ""}
    ${!data.relational_available ? "<p class=\"muted\">Text-only mode — relational tier routing limited.</p>" : ""}
    ${data.case_id ? `<p><strong>Queued as case #${data.case_id}</strong></p>` : ""}
  `;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {
    message: document.getElementById("message").value,
    prefer_full_model: document.getElementById("prefer-full").checked,
    add_to_queue: document.getElementById("add-queue").checked,
  };
  if (useRelational.checked) {
    body.relational = {
      peerness: parseFloat(document.getElementById("peerness").value),
      total_msgs: parseInt(document.getElementById("total_msgs").value, 10),
      aggressive_msgs: parseInt(document.getElementById("aggressive_msgs").value, 10),
    };
  }
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    resultEl.classList.remove("hidden");
    resultEl.innerHTML = `<p class="tier tier-1">Error: ${data.detail || res.statusText}</p>`;
    return;
  }
  renderResult(data);
  loadQueue();
});

async function reviewCase(id, decision) {
  const note = prompt(`Optional note for case #${id}:`) || "";
  await fetch(`/api/review/${id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, note }),
  });
  loadQueue();
}

function renderQueue(cases) {
  queueList.innerHTML = "";
  if (!cases.length) {
    queueEmpty.style.display = "block";
    return;
  }
  queueEmpty.style.display = "none";
  cases.forEach((c) => {
    const li = document.createElement("li");
    li.className = "queue-item";
    li.innerHTML = `
      <div class="meta">Case #${c.id} · ${c.model_id} · ${(c.probability * 100).toFixed(1)}% · Tier ${c.tier}</div>
      <div class="preview">${escapeHtml(c.message_preview)}</div>
      <div class="muted">${escapeHtml(c.tier_reason)}</div>
      <div class="queue-actions">
        <button class="btn small secondary" data-id="${c.id}" data-action="approved">Approve</button>
        <button class="btn small secondary" data-id="${c.id}" data-action="dismissed">Dismiss</button>
        <button class="btn small primary" data-id="${c.id}" data-action="escalated">Escalate</button>
      </div>
    `;
    queueList.appendChild(li);
  });
  queueList.querySelectorAll("button[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => {
      reviewCase(parseInt(btn.dataset.id, 10), btn.dataset.action);
    });
  });
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}

async function loadQueue() {
  const res = await fetch("/api/queue");
  const data = await res.json();
  renderQueue(data.cases || []);
}

document.getElementById("refresh-queue").addEventListener("click", loadQueue);

async function runDissertationCapture() {
  const params = new URLSearchParams(location.search);
  if (params.get("capture") !== "result") return;

  document.getElementById("message").value =
    "If you continue this behaviour I am going to report you and get you blocked from the site.";
  document.getElementById("peerness").value = "0.93";
  document.getElementById("total_msgs").value = "2";
  document.getElementById("aggressive_msgs").value = "1";
  useRelational.checked = true;
  useRelational.dispatchEvent(new Event("change"));
  document.getElementById("prefer-full").checked = true;
  document.getElementById("add-queue").checked = false;

  const layout = document.querySelector(".layout");
  layout.style.gridTemplateColumns = "1fr";
  layout.children[1].style.display = "none";

  const body = {
    message: document.getElementById("message").value,
    prefer_full_model: true,
    add_to_queue: false,
    relational: {
      peerness: 0.93,
      total_msgs: 2,
      aggressive_msgs: 1,
    },
  };
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (res.ok) renderResult(data);
}

checkHealth();
loadQueue();
useRelational.dispatchEvent(new Event("change"));
runDissertationCapture();

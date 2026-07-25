# Cyberbullying Moderator Dashboard (Frontend)

Self-contained **option C** prototype: **FastAPI** backend + **HTML/CSS/JS** UI.  
Implements human-in-the-loop review with three-tier routing (dissertation §4.6).  
**Does not modify** `src/`, notebooks, or dissertation files — only reads trained models from `outputs/models/`.

## Why this stack (vs A / B)

| Option | Choice |
|--------|--------|
| **A** — Dissertation text only | No runnable demo |
| **B** — Streamlit | Fast but blurs UI/API; weaker match to “moderator dashboard” |
| **C** — FastAPI + web UI ✓ | Separate `frontend/` folder, REST API, queue + audit trail, aligns with R5 intervention design |

## Prerequisites

- Trained models at project root:
  - `outputs/models/baseline_a1.joblib` (required)
  - `outputs/models/full_a4.joblib` (optional; enables A4)
- Python 3.10+

## Setup

From the **project root**:

```bash
python3 -m venv frontend/.venv
frontend/.venv/bin/pip install -r frontend/requirements.txt
```

## Run

From the **project root** (so `src/` imports resolve):

```bash
cd frontend
PYTHONPATH=..:. ../.venv/bin/python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8080
```

Or use the project venv if dependencies are already installed:

```bash
cd frontend
PYTHONPATH=..:. ../.venv/bin/uvicorn api.main:app --reload --port 8080
```

Open **http://127.0.0.1:8080**

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Model availability |
| `POST` | `/api/predict` | Classify message (+ optional relational fields) |
| `GET` | `/api/queue` | Pending moderator cases |
| `POST` | `/api/review/{id}` | Approve / dismiss / escalate |
| `GET` | `/api/cases` | Recent case history |

## Modes

- **A4 (full):** message + peerness + TotalMsgs + AggressiveMsgs → tiered queue
- **A1 (text-only):** message only, or when A4 weights missing → lower-trust path

## Demo examples (copy into the dashboard)

The full model (A4) uses **message text and relational fields together**. Text alone often scores low on A4; that matches the dissertation finding that peerness and repetition matter.

**UI tips**

- Leave **Include peerness & repetition** checked for A4 examples.
- Uncheck it for A1 text-only examples.
- After analysis, use **Approve** (agree with flag), **Dismiss** (false alarm), or **Escalate** (needs senior review) in the review queue.

### 1. Benign — should not queue

**Message:**
```text
Thanks for sharing the notes — really helped for the exam.
```

| Peerness | Total messages | Aggressive messages |
|----------|----------------|---------------------|
| 0.55 | 3 | 0 |

**Expected:** Not cyberbullying · no review queue (or Tier 0).

---

### 2. Main A4 demo — likely flagged (Tier 1 / 2)

**Message:**
```text
If you continue this behaviour I am going to report you and get you blocked from the site.
```

| Peerness | Total messages | Aggressive messages |
|----------|----------------|---------------------|
| 0.93 | 2 | 1 |

**Expected:** Model **A4** · score ~80%+ · cyberbullying predicted · added to queue (often **Tier 1** — high `combined_score` + repeat-offender pattern).

---

### 3. Text-only insult — uncheck relational fields

**Message:**
```text
You are pathetic and nobody likes you. Leave me alone.
```

**Settings:** Uncheck **Include peerness & repetition** · prefer **A1**.

**Expected:** Model **A1** · score often ~60%+ · may flag without relational metadata.

**Alternative (A1):**
```text
Nobody wants you here. Go away loser.
```

---

### 4. Offensive text, weak relational context — peerness lesson

**Message:**
```text
You are an idiot and your opinion is worthless.
```

| Peerness | Total messages | Aggressive messages |
|----------|----------------|---------------------|
| 0.25 | 1 | 1 |

**Expected:** Often **not flagged** on A4 — shows that harsh wording alone is not enough without peer/repeat context.

---

### 5. Repeat-harassment wording — domain mismatch demo

**Message:**
```text
Stop messaging me every day. This is harassment and I want you to leave me alone.
```

| Peerness | Total messages | Aggressive messages |
|----------|----------------|---------------------|
| 0.85 | 6 | 4 |

**Expected:** May stay **not flagged** on A4 (wording unlike training corpus) — useful to explain domain shift, not a broken UI.

---

### Review queue actions

| Button | Meaning |
|--------|---------|
| **Approve** | Moderator agrees it is cyberbullying; intervention warranted (logged only in this prototype). |
| **Dismiss** | False alarm or not cyberbullying; close case. |
| **Escalate** | Serious or unclear; send to senior / safeguarding review. |

In production, relational fields would come from the platform (user-pair history), not typed manually. This dashboard uses manual inputs for research demonstration only.

## Data

SQLite database: `frontend/data/moderator_cases.db` (created on first run).

## PLES

This is a **research prototype**. The UI shows disclaimers; all actions require explicit moderator review — no autonomous sanctions.

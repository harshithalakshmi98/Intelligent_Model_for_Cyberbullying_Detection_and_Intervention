# Frontend Demo Script — Simple Walkthrough

**Project:** F21MP · Moderator dashboard prototype (`frontend/`)  
**Full context:** [`supervisor-meeting-script.md`](supervisor-meeting-script.md) (Section 6)  
**Time:** ~3 minutes spoken + live demo

---

## Before you start

**Run the dashboard:**

```bash
cd frontend
PYTHONPATH=..:. ../.venv/bin/uvicorn api.main:app --reload --port 8080
```

Open: **http://127.0.0.1:8080**

**Have ready:** Dissertation Figures 4.5–4.7 (optional, on screen behind you).

---

## 1. What I built (30 seconds)

> “After training the models in Python, I built a **small local website** so a moderator can try the system — not to punish users automatically, but to show **how a human would review flagged cases**.
>
> It loads my saved models (**A1** = text only, **A4** = text + relationship + behaviour). When I submit a message, the backend scores it, picks a **review tier** (urgent / standard / monitor), and puts serious cases in a **queue**. A person must **Approve**, **Dismiss**, or **Escalate** — the machine only recommends.”

**One line:** Training code lives in `src/`; this `frontend/` folder is a **proof-of-concept** for the intervention workflow (dissertation Section 4.9).

---

## 2. Why I type peerness and message counts (1 minute)

> “You might notice I’m typing numbers as well as the message. That’s **on purpose**, not bad UI design.
>
> My full model **A4** was trained on more than words. On the Towards Comprehensive dataset, cyberbullying is linked to:
> - **Peerness** — how close the two users are (e.g. classmates vs strangers)  
> - **Repetition** — how many messages they exchanged and how many were aggressive  
>
> So the form asks for:
> - **Peerness** (0 to 1)  
> - **Total messages** between the pair  
> - **Aggressive messages** in that thread  
>
> From those, the system works out things like **aggression ratio** and **combined score**, which decide whether a case is **Tier 1 — urgent** or lower priority.
>
> **If I leave those fields out**, it falls back to **A1 — text only** — which my results show is **very weak** on this dataset (F1 = 0).
>
> **In a real app**, a moderator would **not** type these. Twitter or a school platform would already know user pairs and message history. I type them here because this is a **research demo** on public data, not a live network.”

---

## 3. Live demo — what to say and what to click (2 minutes)

### Demo A — Normal message (should not alarm)

**Say:**

> “First, a harmless message between people who aren’t in a hostile pattern.”

**Type:**

| Field | Value |
|-------|-------|
| Message | `Thanks for sharing the notes — really helped for the exam.` |
| Peerness | `0.55` |
| Total messages | `3` |
| Aggressive messages | `0` |
| Include peerness & repetition | **Checked** |

**Say after Submit:**

> “**Not cyberbullying** — nothing urgent in the queue. That’s what we want for normal chat.”

---

### Demo B — Main point (A4 flags + queue)

**Say:**

> “Same kind of demo as in my results: harsh message **plus** high peerness and a repeat pattern — what A4 was trained on.”

**Type:**

| Field | Value |
|-------|-------|
| Message | `If you continue this behaviour I am going to report you and get you blocked from the site.` |
| Peerness | `0.93` |
| Total messages | `2` |
| Aggressive messages | `1` |
| Include peerness & repetition | **Checked** |

**Say after Submit:**

> “Model **A4** runs. High score — about **80%**. Predicted **cyberbullying**. **Tier 1 — urgent review**. Case appears in the **queue** on the right.
>
> This is the thesis in one screen: **words alone aren’t enough on this corpus; relationship and repetition matter.**”

**Click one action and say:**

> “I’m the moderator. I can **Approve** if I agree it’s bullying, **Dismiss** if it’s a false alarm, or **Escalate** for safeguarding. Nothing happens to a real user — it’s logged for audit only.”

---

### Demo C — Optional: text-only path (30 seconds)

**Say:**

> “If I **uncheck** peerness and repetition, only **A1** runs — text-only, like many published systems.”

**Type:** any insulting message (e.g. `You are pathetic and nobody likes you.`)

**Uncheck:** Include peerness & repetition

**Say:**

> “It may still flag sometimes, but my **Phase 1** result was F1 **zero** on the test set without relational features. This shows why I don’t claim A4 works on Twitter without real pair data.”

---

## 4. Close the demo (20 seconds)

> “So the dashboard doesn’t add a second research result — it **shows requirement R5 in practice**: tiered review, human in the loop, no automatic blocking. Production would plug in platform APIs; this prototype proves the **workflow** is workable.”

---

## Quick reference — fields on the form

| Field | Simple meaning | Why A4 needs it |
|-------|----------------|-----------------|
| **Message** | What was said | Text features (TF-IDF) |
| **Peerness** | How close are the two users? | Biggest gain in ablation (+0.65 F1) |
| **Total messages** | How much back-and-forth? | Repetition / pattern |
| **Aggressive messages** | How many were already hostile? | `aggression_ratio`, repeat-offender flag |

---

## If they ask one question

| Question | Short answer |
|----------|----------------|
| Is this deployed? | No — local research prototype only. |
| Why manual typing? | No live platform API; public dataset demo. |
| What does the frontend prove? | Moderators can review tiered flags with audit log — human-in-the-loop. |

---

*More detail: `frontend/README.md` · dissertation Section 4.9*

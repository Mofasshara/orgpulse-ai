# OrgPulse AI — PharmaOps: Learning Guide

> A living guide that explains every concept, tool, and decision in this project —
> what it is, why it exists, and how it works — in plain English first, then
> mechanism. Written for an architect growing toward a developer's understanding.
>
> **Scope note:** This guide grows as we build. Three concepts were prioritized
> for deep coverage — **Hermes autonomy, LangGraph, and Prophet
> backtesting/calibration** — because they're the least familiar and the most
> impressive to explain in an interview. Other components (generators, anomaly
> detectors, the audit log) get entries as they're built.
> **Last updated:** 2026-08-03

---

## The Big Picture

Most business dashboards are **passive**: they show you what already happened and
wait for a human to notice a problem. OrgPulse is the opposite — a **standing
agent** for a pharmaceutical operations team. On a schedule (say, 7am daily) it
wakes up on its own, generates the latest (synthetic) numbers for three parts of
the business (clinical trials, the QC lab, and the supply chain), works out what
looks wrong today, predicts where the hero metric (clinical enrollment) is heading
over the next 30/60/90 days, and writes a plain-English recommendation phrased the
way a pharma quality team expects — all before anyone has opened their laptop.
Because pharma is heavily regulated, every recommendation is logged in a
**tamper-evident** trail (under the threat model in the case study — it catches
corruption and naïve edits, not a fully re-signed forgery) and marked "needs a
human to sign off" — the AI advises, it never decides. (This prototype uses
realistic *synthetic* data, not a live business feed.)

In one line: *it's the architecture of Palantir AIP — ingest → detect → forecast
→ recommend — but open-source, self-hosted, and built to the EU AI Act
high-risk + GxP bar.*

---

## System Architecture

```
                          ┌──────────────────────────────────────────┐
                          │  HERMES  (the autonomic nervous system)   │
                          │  • wakes on a cron (default 07:00 daily)  │
                          │  • remembers last period's baselines      │
                          │  • decides when to alert                  │
                          └───────────────────┬──────────────────────┘
                                              │ invokes once per run
                                              ▼
   3 synthetic data streams          ┌─────────────────────────────────┐
   ┌───────────────────┐             │  LANGGRAPH DAG (the reasoning)  │
   │ Clinical trials ──┼────────────▶│  1. Anomaly Detector            │
   │ QC / lab        ──┼────────────▶│  2. Forecaster (Prophet)        │
   │ Supply chain    ──┼────────────▶│  3. Domain Analyst   (LLM)      │
   └───────────────────┘             │  4. Recommendation Writer (LLM) │
                                      └───────────────┬─────────────────┘
                                                      │ produces
                                 ┌────────────────────┼─────────────────────┐
                                 ▼                    ▼                     ▼
                        findings.json          report.md            hash-chained
                        (structured)        (plain language)        audit log entry
                                 │                    │                     │
                                 └──────────┬─────────┴─────────────────────┘
                                            ▼
                              runs/<timestamp>/  +  runs/latest/
                                            │
                                            ▼
                          ┌──────────────────────────────────────┐
                          │ STREAMLIT dashboard (the shop window) │
                          │ reads runs/latest, renders it,        │
                          │ deploy-ready (not yet deployed)       │
                          └──────────────────────────────────────┘
```

**Why two "brains" (Hermes AND LangGraph)?** Hermes decides *when to act and what
to remember* across time. LangGraph decides *what to think* within a single run.
Keeping them separate means each is simple and independently testable — and it
mirrors exactly how Palantir splits "Agent Studio" (the autonomous monitor) from
"AIP Logic" (the analysis).

---

## Part 1 — The Autonomic Loop

*This layer's job: run the whole system on a schedule, with a memory, without a
human pressing "go".*

### Hermes — the autonomic nervous system

**What it is**
Hermes is like a **night-shift security guard who does rounds**. Nobody tells the
guard to start each patrol — a schedule does. The guard carries a notebook from
previous nights ("the east door is normally locked by 6pm"), so they notice when
tonight is different, and they raise an alarm only when something crosses a line.
Technically, Hermes (by Nous Research) is an autonomous AI agent framework that
can run on a **cron** (a time-based scheduler), keep **persistent memory** between
runs, and call **tools** (other programs) to get work done. In OrgPulse, Hermes is
the outer loop: it wakes up, loads yesterday's baselines, triggers the analysis,
saves the new baselines, writes the results, and fires an alert if needed.

**The problem it solves**
Without an autonomous scheduler-with-memory, someone has to *remember* to run the
analysis every morning, and the system has no sense of "normal" to compare
against. The specific failure mode: a dashboard that only updates when a human
clicks refresh, and that treats today's data in isolation with no memory of what
last week looked like — so it can't tell "300 enrollments" is a crash if last week
was 900. Hermes removes the human trigger and gives the system a memory.

**How it works**
1. A cron entry tells Hermes "run this task at 07:00 every day."
2. On waking, Hermes reads its **memory store** (`~/.hermes/memories/`) to recover
   the previous run's baseline metrics per KPI.
3. Hermes runs the three domain **SKILL.md** instructions — small Markdown files
   that tell it how to monitor each stream — which ingest that day's data.
4. Hermes calls the analysis pipeline (the LangGraph DAG) as a **tool** and gets
   back a structured findings object.
5. It writes the new metrics back to memory as the next baseline, saves the run
   artifacts to disk, and emits an alert if any KPI is "Critical".

| Piece | Plain-English role |
|-------|--------------------|
| cron | The alarm clock that starts each run |
| memory store | The notebook of what "normal" looked like last time |
| SKILL.md | A written standard-operating-procedure per stream |
| tool call | Handing the actual analysis to the LangGraph pipeline |

**Why we chose it for this project**
The brief requires a *genuinely autonomous* agent, and Hermes was already
installed and is purpose-built for scheduled, memory-backed, tool-using autonomy.
The alternative considered was a plain **cron job running a Python script** — far
simpler, but it has no concept of agent memory, no tool-orchestration model, and
tells a much weaker story ("I wrote a shell script" vs. "I deployed an autonomous
agent"). We gave up some simplicity for a real agent runtime and a stronger
portfolio narrative. A second alternative — **Airflow/Prefect** (data-pipeline
schedulers) — was rejected because they orchestrate *data tasks*, not an
*autonomous reasoning agent*, which is the thing we're showcasing.

**Key things to know**
- Hermes cannot live on Streamlit Cloud (that host sleeps and is stateless) — this
  is *why* the system is "split-plane": Hermes runs as the engine locally and
  publishes artifacts that the cloud dashboard merely reads.
- The memory is the single source of "what was normal" — if it's wiped, the first
  run afterward has no baseline and can only report levels, not changes.

---

## Part 2 — The Analysis DAG

*This layer's job: within one run, turn today's raw numbers into anomalies,
forecasts, and a written recommendation.*

### LangGraph — the conductor of the analysis

**What it is**
LangGraph is **the conductor of an orchestra**. The conductor plays no instrument
— they decide who plays, in what order, and pass the score along. Technically,
LangGraph is a framework for building **stateful, multi-step AI workflows** as a
**graph**: each node is a step, edges define the order, and a shared **state**
object is passed from node to node, each one adding to it. OrgPulse's analysis is
a 4-node chain: Anomaly Detector → Forecaster → Domain Analyst → Recommendation
Writer. The first two are pure math/ML; the last two are LLM calls.

**The problem it solves**
You *could* write these four steps as one long function. The failure mode that
causes: it becomes a tangled block that's impossible to test in isolation, hard to
re-run a single step, and gives you no clean place to inspect the intermediate
state (the anomalies before they became a recommendation). LangGraph forces each
step into a named, independently-testable node with an explicit shared state —
so you can unit-test the detector without the LLM, swap the forecaster, or replay
a run and see exactly what each stage produced.

**How it works**
1. Define a **state schema** — a structured object holding the run's data,
   detected anomalies, forecasts, and recommendations (starts mostly empty).
2. Define four **nodes**, each a function that reads the state, does its job, and
   returns the fields it fills in.
3. Wire the **edges**: detector → forecaster → analyst → writer.
4. **Invoke** the graph with the initial state (today's data + the baseline); it
   runs each node in order and returns the fully-populated state.

```
state = {data, baseline}
   │  detector adds  → anomalies
   │  forecaster adds → forecasts
   │  analyst adds    → interpretation
   │  writer adds     → recommendations
   ▼
final state → findings.json
```

**Why we chose it for this project**
The pipeline is inherently multi-step with a mix of deterministic (math) and
non-deterministic (LLM) stages, and we need each stage testable and inspectable —
exactly LangGraph's sweet spot. Alternatives considered: (1) **a plain function
chain** — simplest, but loses per-step isolation, state inspection, and the
ability to later add branching (e.g. skip the LLM in offline mode) cleanly; (2)
**CrewAI/AutoGen** (multi-*agent* frameworks) — rejected because our four steps
are a fixed pipeline, not autonomous agents negotiating with each other; that
framing would add conversational overhead we don't need. We gave up a little
boilerplate for structure and testability.

**Key things to know**
- The state object is the contract between nodes — if the detector renames a
  field, the forecaster must agree. Keeping the schema explicit prevents this.
- Only nodes 3 and 4 need an LLM (and an API key). In **offline mode** node 4 is
  swapped for a deterministic template, so the whole graph still runs keyless —
  this is what keeps the public demo alive without a paid key.

### Prophet — the forecasting engine (with honest scoring)

**What it is**
Prophet is a **weather forecaster for business metrics**. A weather forecaster
doesn't just say "it'll be 20°C" — they say "20°C, 80% chance between 17 and 23",
and good ones are judged on whether reality lands in their range. Prophet (from
Meta) is a time-series forecasting library that, given a history of daily values,
predicts future values *with a confidence band* (not just a single line). But the
important part of this project isn't calling Prophet — it's **proving the forecast
is trustworthy** through backtesting and calibration.

**The problem it solves**
Anyone can call `Prophet.fit()` and draw a pretty line. The failure mode that
hides: the line can be confidently wrong, and on data you generated yourself, a
good-looking fit proves *nothing* (you already knew the answer). Two techniques
fix this: **backtesting** (test the forecast on data it never saw) and
**calibration** (check the confidence band is honest).

**How it works — the two techniques that make it rigorous**
1. **Rolling-origin backtest:** pretend it's an earlier date, hide everything
   after it, forecast forward, then reveal the truth and measure the error. Slide
   this "origin" forward repeatedly and average the errors.
   - **MAPE** (Mean Absolute Percentage Error): "on average the forecast was off
     by X%." Easy to say to a stakeholder.
   - **MASE** (Mean Absolute Scaled Error): the error compared to a dumb
     "tomorrow = today" guess. **Below 1.0 = smarter than the naive guess**;
     above 1.0 = worse than doing nothing clever. This is the honest yardstick.
2. **Interval calibration:** collect all the "90% confidence" predictions and
   check whether reality actually fell inside the band ~90% of the time. If your
   "90% band" only catches 60% of real values, it's lying — the model is
   overconfident. A calibrated model's 90% band catches ~90%.

```
history ──▶ [hide the future] ──▶ forecast ──▶ reveal truth ──▶ error (MAPE/MASE)
                     ▲                                              │
                     └──────── slide origin forward, repeat ◀───────┘
```

**Why we chose it for this project**
Prophet handles the messy realities of our clinical-enrollment data — seasonality,
holiday dips, trend changes — out of the box, and natively produces the confidence
intervals we need. Alternatives considered: (1) **statsforecast (AutoARIMA/ETS)** —
excellent and lighter to install; kept as a fallback if Prophet's heavy compiled
dependency causes deployment friction; (2) **a deep-learning forecaster
(LSTM/N-BEATS)** — rejected as overkill for daily univariate KPIs and far harder
to explain to a regulator (interpretability matters in pharma). We chose the tool
whose forecasts we can *defend*, not just produce.

**Key things to know**
- Prophet pulls a heavy compiled dependency (cmdstanpy) — this is a known
  deployment risk (see ROADMAP risk R1); the lightweight Streamlit app deliberately
  does *not* depend on it (it only reads pre-computed forecasts).
- Forecasting on self-generated data is only credible *because* we report
  held-out backtest metrics and calibration — never in-sample fit.

---

## Part 3 — The Presentation Plane

*This layer's job: show the world what the agent found, cheaply and safely, without
ever touching the machinery that produced it.*

### The split-plane pattern — the engine room vs. the shop window

**What it is**
Think of a **restaurant**: there's a kitchen (hot, dangerous, full of equipment and
raw ingredients) and a dining room (calm, pleasant, where guests actually sit). You
would never seat guests in the kitchen. OrgPulse is built the same way. The
**compute plane** is the kitchen — the Hermes agent running Prophet, LangGraph, and
scikit-learn locally to cook up results. The **presentation plane** is the dining
room — the Streamlit dashboard where anyone can see those results plated up. The only
thing passed between them is finished plates: the committed files in `runs/`.

**The problem it solves**
If the dashboard *were* the kitchen — if it imported Prophet and ran the model on
demand — three things break: it becomes **expensive and fragile to host** (Prophet's
compiled dependency often fails to install on free hosts), it needs to **hold an API
key** in a public place, and it's **slow or dead on first load** while a model spins
up. Splitting the planes makes the public surface a thin, keyless, always-populated
view that renders pre-computed files.

**How it works**
1. The Hermes agent runs a tick locally, does all the heavy computation, and writes
   three small files per run into `runs/` (and mirrors the newest into `runs/latest`).
2. Those files are **committed to git** — so they're already present when the app
   deploys. The dashboard is alive on first load with no computation.
3. The Streamlit app reads `runs/latest`, renders it, and **re-verifies the audit
   chain** — but imports *nothing* from the `orgpulse` package. Its dependency list is
   three lightweight libraries.

```
KITCHEN (compute plane)          plated dishes         DINING ROOM (presentation plane)
Hermes + Prophet + LangGraph ──▶  runs/*.json  ──▶  Streamlit dashboard (streamlit/pandas/plotly)
(heavy, local, holds the key)     (committed)        (light, hosted, keyless, read-only)
```

**Why we chose it for this project**
The alternative — a single Streamlit app that runs the model live — is simpler to
wire up but fails exactly where a portfolio demo must not: on the public URL. We also
considered a **separate backend API** (FastAPI serving results to the dashboard), but
that adds a server to host and monitor for no benefit here; committed files *are* the
API. The split-plane choice buys a demo that is always up and costs nothing to run.

### Streamlit — the shop window

**What it is**
Streamlit is a **"turn a Python script into a web page" machine**. Normally, building
a web dashboard means HTML, CSS, JavaScript, and a server. Streamlit lets you write
ordinary top-to-bottom Python (`st.title(...)`, `st.metric(...)`, `st.dataframe(...)`)
and it renders each statement as a piece of a live web app. Plotly, used alongside it,
is the **charting library** that draws the interactive forecast fan (the shaded
uncertainty band you can hover over).

**The problem it solves**
A dashboard is how a non-technical stakeholder — a quality manager, an auditor — sees
what the agent found without reading JSON. Without something like Streamlit, showing
the work means either raw files (unreadable to humans) or a full front-end project
(weeks of work). Streamlit compresses "make it viewable by a human" from weeks to an
afternoon.

**How it works**
1. The app runs as a Python script, top to bottom, on every page load.
2. Each `st.*` call emits a widget: a title, a metric card, a coloured severity chip,
   a Plotly chart, a dataframe.
3. It loads the committed `findings.json` / `audit.jsonl`, colours each KPI by
   severity, draws the forecast fan, lists the human-gated recommendations, and shows
   a live ✅/❌ on the audit chain.

**Why we chose it for this project**
Streamlit deploys to **Streamlit Community Cloud** for free straight from a GitHub
repo — ideal for a public portfolio piece. Alternatives: (1) **Dash/Plotly** — more
powerful and customizable, but more boilerplate than this needs; (2) a **static site
+ pre-rendered charts** — cheapest to host but loses interactivity (hovering the
forecast band, sortable audit table). Streamlit is the sweet spot of "fast to build,
free to host, interactive enough."

**Key things to know**
- The app imports **no `orgpulse` code** on purpose — that's what keeps its
  dependency list to three libraries and the split plane intact. It even
  re-implements the audit-chain check standalone, with a test proving it agrees with
  the real one (so a green checkmark on the dashboard means the same thing the compute
  plane means by it).
- Everything the dashboard shows is a **committed artifact**, so the demo is
  reproducible: `scripts/seed_runs.py` regenerates the exact same run history
  byte-for-byte.

---

## Key Terms Glossary

| Term | Plain-English meaning |
|------|-----------------------|
| Agent (autonomous) | Software that acts on its own schedule and initiative, not only when asked |
| Cron | A time-based scheduler ("run this at 07:00 daily") |
| Baseline | The remembered "normal" level of a metric, used to judge whether today is off |
| SKILL.md | A Markdown standard-operating-procedure telling Hermes how to monitor a stream |
| Tool call | An agent handing a job to another program and getting the result back |
| DAG | Directed Acyclic Graph — steps connected in a one-way order with no loops |
| Node / Edge | A single step / the arrow defining which step runs next |
| State (LangGraph) | The shared object passed between steps, each step adding to it |
| LLM | Large Language Model — the AI that writes the plain-English analysis |
| Time series | A metric measured repeatedly over time (daily enrollment, etc.) |
| Confidence interval | The range a forecast is fairly sure the true value will fall in |
| Backtest | Testing a forecast on past data it wasn't allowed to see |
| MAPE | Average forecast error as a percentage |
| MASE | Forecast error vs. a naive "tomorrow = today" guess; <1.0 is good |
| Calibration | Checking a "90% confident" band really is right ~90% of the time |
| Split-plane | Compute (Hermes, local) separated from presentation (Streamlit Cloud) |
| CUSUM | A control-chart method that detects slow drifts/shifts in a metric |
| GxP / ICH E6(R3) | Pharma "good practice" quality standards for regulated work |
| EU AI Act (high-risk) | EU law imposing logging, oversight, and documentation duties on risky AI |

---

## How to Explain This in an Interview

**60-second version (corridor / first meeting)**
"OrgPulse is an autonomous operations analyst for pharma. Instead of a dashboard
you have to check, an AI agent wakes up on a schedule, ingests data from clinical
trials, the QC lab, and the supply chain, spots what's abnormal, forecasts the
hero KPI over the next 30/60/90 days, and writes a recommendation in the language a
quality team uses — and because it's pharma, every recommendation is logged
tamper-evidently and
needs human sign-off. It's Palantir AIP's shape, but open-source and built to the
EU AI Act bar."

**Technical version (senior engineer)**
"Two-tier orchestration. Hermes is the outer autonomic loop — cron-triggered,
holds baselines in persistent memory across runs, and invokes the pipeline as a
tool. The inner pipeline is a LangGraph DAG with a shared state object: anomaly
detection (window-calibrated CUSUM + Z-score against the remembered baseline;
an Isolation Forest detector is built and tested but not yet wired),
Prophet forecasting with rolling-origin backtests reporting MASE and an in-sample
interval calibration applied to the published band, then two LLM nodes for domain
interpretation and a CAPA-framed
recommendation. Governance is a hash-chained append-only audit log that records the
input lineage (seed, parameters, input-data hash, and the remembered-baseline hash)
plus output-integrity hashes, so an offline decision can be exactly regenerated (and any
run's outputs checked) per EU AI
Act Article 12, with human-oversight gating per Article 14. Deployment is split-plane:
Hermes computes locally and publishes run artifacts; a lightweight Streamlit app on
Streamlit Cloud just reads the latest run, so the heavy forecasting stack never has to
deploy to the cloud."

**Business version (executive / regulator)**
"It's a proactive assistant for operations leaders. Today, problems in trial
enrollment or supply are often noticed late, after they've already cost time or
money. This system watches continuously, warns early with a forecast of where
things are heading, and tells the team what to consider doing — always as advice a
qualified person must approve, never an automated decision. It's designed from the
ground up for a regulated environment: every suggestion is recorded in a
tamper-evident log with its full input lineage, so an auditor can trace it and — for
the deterministic offline runs — reproduce it exactly, in line with the EU AI Act's
rules for high-risk AI."

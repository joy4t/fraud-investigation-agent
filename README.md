# Fraud Investigation Agent

An autonomous AI agent that investigates flagged credit card transactions and returns a structured risk assessment a fraud analyst can act on. Built with LangGraph on Llama 3.3 70B, deployed as a FastAPI service on Railway.

**Live API:** https://fraud-investigation-agent-production.up.railway.app  
**Interactive docs:** https://fraud-investigation-agent-production.up.railway.app/docs

---

## Problem

Banks flag far more transactions than human analysts can investigate. A classical ML model can produce a fraud probability, but a probability score isn't an investigation — it doesn't tell the analyst *why*, and it doesn't produce a report that can be acted on or audited.

What this project does: turns a flagged transaction into a structured, auditable investigation, autonomously.

## Solution

An AI agent that takes a transaction ID as input and returns a five-section fraud report. The agent orchestrates four specialized tools:

| Tool | What it does |
|------|--------------|
| `transaction_inspector` | Returns raw facts for the transaction — amount, time, merchant, category, distance from home |
| `customer_profiler` | Builds a behavioral baseline from the customer's transaction history — typical amount, hours, categories, home location |
| `risk_scorer` | Weighs four signals (amount deviation, time anomaly, distance, category mismatch) into a 0-100 score with risk level |
| `report_generator` | Assembles a structured five-section report with verdict, evidence, and recommended action |

Tools return evidence, not verdicts — reasoning happens at the agent layer.

**On sequencing:** the LangGraph architecture supports dynamic tool selection — the LLM can call any tool in any order based on state. For v1, I deliberately constrained the agent with a prescribed investigation sequence in the system prompt (inspector → profiler → scorer → reporter). An auditable, reproducible investigation path matters more in a compliance domain than agent autonomy for its own sake. Freeing the sequencing is a v2 experiment tied to adding branch-point tools.

**Why an agent at all, not a rules engine or a classical ML model:** rules engines don't handle novel patterns and go stale as fraudsters adapt. Classical ML gives you a score with no narrative, which a human analyst can't act on. The agent architecture produces evidence-weighted reasoning in natural language and, critically, is *extensible* — new investigation capabilities are added by writing new tools, no retraining. In production I'd pair this with a classical ML pre-filter: cheap model flags the top 5% suspicious transactions, the agent investigates those.

## Impact

- **Live API in production** on Railway with auto-generated Swagger docs.
- **Deterministic outputs** — temperature-zero LLM produces the same verdict on the same transaction every time, meeting the audit bar for regulated industries.
- **Structured, explainable output** — every risk score comes with a per-signal breakdown, so an analyst can see exactly why a transaction scored what it did (e.g., *"10 points because amount was 2σ above median, 25 points because time was 5 hours outside typical range"*).
- **Modular tool design** — each tool has a single responsibility. New investigation dimensions can be added by writing new tools without touching the agent core.

---

The graph is minimal on purpose: two nodes, one conditional edge, one loop. Chaining across tool calls happens via an accumulating message list in state — every tool result becomes context for the LLM's next decision. New tools can be added by appending to the tool list and updating the system prompt; no graph rewiring required.

**Stack:** Python 3.11 · LangGraph · LangChain · Llama 3.3 70B (Meta, open-weights) served via Groq's inference API · FastAPI · Pandas · Railway.

**Data:** Kaggle credit card fraud dataset (`kartik2112/fraud-detection`), sampled to 50K rows. Sampling was necessary because GitHub's 100MB per-file limit ruled out the full 1.3M-row original; sampling preserves the same customer/merchant distributions and reproducibility is guaranteed via `random_state=42`.

---

## Design decisions worth naming

**Temperature zero.** Fraud investigation is a regulated workflow — the same transaction must produce the same risk assessment across runs. Any non-determinism fails audit. Temperature zero means the LLM picks the highest-probability tool call every time; same input, same output.

**Tools return evidence, not verdicts.** A tool that returned `is_suspicious: true` would encode a threshold decision inside the tool that only makes sense in context of the customer's baseline — which lives in a different tool. Instead, tools return raw facts (distance in km, amount vs median, hour vs typical range). The agent weighs them together. Reasoning stays at the agent layer, where it can be traced.

**Data cleaning at the code layer, not in the CSV.** The raw Kaggle dataset labels fraud merchants with a `fraud_` prefix in the merchant name — a target leak. The cleanup happens in the transaction inspector via `.replace('fraud_', '')`, not by editing the CSV. Anyone cloning the repo with the original dataset gets identical results. Reproducibility over convenience.

**Heuristic scorer, not a trained model.** The four risk signals are weighted by hand — z-score bands for amount, hour bands for time, kilometer bands for distance, categorical match for category. Two reasons: (1) the labeled sample is under 300 fraud cases, too thin for reliable supervised learning; (2) explainability is a compliance requirement in finance — a rule-based scorer explains itself by construction, a trained model produces a number with no narrative. The tradeoff is deliberate: interpretability over accuracy at this scale. See failure modes below for what this means for the weights.

**FastAPI, not Streamlit.** The intended consumer is a programmatic system (a bank's transaction pipeline calling per transaction), not a human clicking through a UI. FastAPI exposes the agent as a JSON API. Auto-generated Swagger docs at `/docs` let a human still interact with it during development.

---

## How this system fails (and what I've done about it)

An investigation agent that pretends it has no failure modes is more dangerous than one that names them. What follows are known limitations, ranked by how much they'd hurt a production deployment.

**Weights are hand-set, never calibrated against the label.** The four scoring bands were picked by intuition (z<1: 0 pts, z<2: 10 pts, z<3: 20 pts, z≥3: 25 pts, and similar for time/distance/category). I have not measured how well these bands correlate with the `is_fraud` label in the dataset. In principle, a simple logistic regression on the same four features would tell me whether my weights are directionally right and, if so, how far off the magnitudes are. This is planned as part of the eval work below — until it's done, the scorer is defensible as a *heuristic baseline*, not as a calibrated model.

**Baseline contamination.** The customer profiler builds each customer's baseline from their *entire* transaction history — including any past fraud transactions. If a customer has been defrauded before, those fraud transactions get treated as "normal for this customer" and inflate the baseline's variance, which desensitizes the scorer to future fraud on that account. Real production must exclude confirmed-fraud rows from the baseline. *Planned fix:* filter `is_fraud == 0` when constructing the baseline. Cheap fix, real bug.

**Cold start on new customers.** A customer with fewer than ~30 transactions has a statistically meaningless baseline — median and standard deviation from three data points are noise, and one outlier dominates. The scorer will fire spurious signals on almost every transaction from such customers. *Planned fix:* minimum-history threshold. Below it, return "insufficient history — manual review" instead of a score.

**Circular time-of-day math.** The customer profiler computes typical activity hours via 25th/75th percentile on the hour axis (0–23). This treats time as linear, but time is circular — hour 23 and hour 0 are adjacent, not 23 hours apart. A night-shift worker who transacts between 10 PM and 4 AM gets an IQR of roughly `01–22`, making their normal window look like nearly the full day, and the time-anomaly signal silently dies for that segment. *Planned fix:* rotate the hour axis per customer to place the inactivity gap at the seam, or use a sin/cos embedding for true circular statistics.

**Evaluation is spot-checking, not measurement.** The system is currently sanity-checked on hand-picked fraud and legit cases. This is not evaluation. Real evaluation for an agent needs two layers: (1) *output evaluation* — labeled test set with precision, recall, F1 on the risk-level classification; (2) *trajectory evaluation* — whether the LLM called the right tools in the right order (an agent can produce the right answer for the wrong reasons, which is still a production bug — this matters even more once the fixed tool sequence is relaxed). *Planned:* 20-case labeled eval harness with both metrics before any weight tuning is considered defensible.

**Internal duplication in `risk_scorer`.** The risk scorer independently computes distance, median amount, standard deviation, and hour percentiles rather than calling the inspector and profiler tools it depends on. Today the numbers are consistent because the scorer is the sole computer of risk scores, but if the profiler is ever tuned (e.g., the circular-stats fix or baseline-contamination fix above), the scorer won't inherit the tuning — silent drift. *Planned fix:* refactor `risk_scorer` to call `transaction_inspector.invoke()` and `customer_profiler.invoke()` for its evidence extraction. Same pattern the `report_generator` already uses.

**Static bands in the scorer.** Distance bands (50/200/500 km) are the same for every customer — a rural retiree who never leaves town and an urban salesperson who travels weekly get the same threshold. Real production would tune bands per customer segment (urban vs rural, travel vs stationary profession). Current bands are defensible defaults, not learned truth.

---

## How I'd measure production success

If this system went into a bank, four metrics — two ML, two business:

- **Precision** (of transactions flagged, % that were actually fraud) — measures customer annoyance from false positives.
- **Recall** (of all fraud, % caught) — measures fraud loss from false negatives.
- **Analyst time per investigation** — a system that flags 100 cases each taking an hour is worse than one flagging 200 that each take five minutes.
- **Cost per fraud dollar prevented** — dollars saved by catching fraud minus dollars spent on false-positive customer service calls and lost trust.

Precision and recall trade against each other and are tuned to the bank's risk appetite. But the business only ultimately cares about the last two — those are downstream of the first two, so all four get measured.

---

## Roadmap

1. **Evaluation harness** (highest leverage — until it exists, every tuning decision is guesswork).
2. **Baseline contamination fix** (`is_fraud == 0` filter in profiler — cheap, real bug).
3. **Weight calibration** against the label via a simple logistic regression, once (1) is in place.
4. **Circular-statistics fix** for time-of-day (silent false-negative bug on night-shift segments).
5. **SSOT refactor** of `risk_scorer` to call inspector/profiler via `.invoke()` (removes drift risk).
6. **Cold-start handling** with minimum-history threshold.
7. **Branch-point tools** (velocity, merchant reputation) to justify freeing the tool sequence and get real agent-ness out of the architecture.

---

## Running locally

```bash
conda create -n fraud-agent python=3.11
conda activate fraud-agent
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
python -m src.api
```

Then hit `http://localhost:8000/docs` for the Swagger UI.






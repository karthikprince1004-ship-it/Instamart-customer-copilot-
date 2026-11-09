# Customer Intelligence & Support Copilot — Teaching Kit

# Overview

### Problem Statement
Quick-commerce platforms like Swiggy Instamart face three core operational challenges as order volume scales: dirty transactional data skewing financial and operational analytics, rising customer churn without proactive retention triggers, and support desk bottlenecks from repetitive refund requests and unmasked customer personal data.

### Solution Description
The **Instamart Customer Intelligence & Support Copilot** is a unified, offline-reproducible system built across four connected components:
1. **Data Foundation**: Cleans raw customer, product, and order records, populates an analytical SQLite database, and runs verification SQL queries.
2. **Customer Segmentation & Lapse Prediction**: Profiles customers using RFM clustering ($K=4$) and trains machine learning models to identify at-risk customers.
3. **Guardrailed Ticket Triage**: Sanitizes customer support tickets (PII masking), handles transient errors, and routes issues based on customer segment and urgency.
4. **Policy RAG & Autonomous Refund Agent**: Retrieves policy facts using TF-IDF cosine similarity and executes a multi-step ReAct agent to process refund requests with strict safety boundaries and state persistence.

### Code Structure
```text
├── generate_data.py          # Deterministic synthetic data generator (unmodified)
├── mock_llm.py               # Deterministic local LLM stand-in (unmodified)
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore patterns (.venv, __pycache__, state/)
├── data_pipeline.py          # Part A: Audit, clean, SQLite storage, SQL queries
├── src/
│   ├── customer_models.py    # Part B: RFM, K-Means clustering, lapse ML models
│   ├── triage.py             # Part C: PII redaction, few-shot triage, token budget
│   ├── policy_rag.py         # Part D: Policy chunking & TF-IDF retrieval
│   └── refund_agent.py       # Part D: ReAct agent, safety stops, crash-resume
├── sql/
│   └── queries.sql           # Part A: Analytical SQL queries
├── data/
│   ├── raw/                  # Generated raw CSVs and policy texts
│   ├── clean/                # Cleaned CSVs (customers, products, orders)
│   └── instamart.db          # Analytical SQLite database
├── state/                    # Part D: Agent execution checkpoints
└── outputs/                  # Exported segment, model, triage, and agent runs
```

### Required Packages & Dependencies
- `pandas` (>= 2.0.0): Data cleaning, manipulation, and CSV persistence
- `numpy` (>= 1.24.0): Vectorized transformations and numerical arrays
- `scikit-learn` (>= 1.3.0): K-Means, PCA, StandardScaler, LogisticRegression, RandomForest
- `joblib` (>= 1.3.0): Model serialization and persistence
- Standard library modules: `sqlite3`, `csv`, `json`, `re`, `math`, `time`, `os`, `sys`

---

## Reproduce

Execute these numbered commands from your terminal:

```bash
# 1. Create and activate an isolated virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1

# 2. Install all required dependencies
pip install pandas numpy scikit-learn joblib

# 3. Generate deterministic synthetic data (1,200 customers, 13,084 raw orders)
python generate_data.py

# 4. Run Part A: Data cleaning, SQLite storage, and SQL verification
python data_pipeline.py

# 5. Run Part B: RFM clustering, PCA, and customer lapse model training
python src/customer_models.py

# 6. Run Part C: Ticket PII redaction, triage evaluation, and segment routing
python src/triage.py

# 7. Run Part D: Policy RAG evaluation and autonomous refund agent
python src/policy_rag.py
python src/refund_agent.py
```

---

## Part A

### Data Audit & Cleaning
- **Raw Audit**: 13,084 order rows (119 exact duplicates, 656 pending records, 12,309 unique `order_id`s, 120 unparseable amounts, 106 negative delivery minutes, 2,541 blank delivery times). Customers: 61 missing ages, 6 impossible ages outside [10, 100], 24 raw city spellings, 10 gender spellings.
- **Cleaning Applied**:
  1. *Customers*: Normalized city spellings to 6 metro cities. Standardized genders to `male` (560), `female` (610), or `unknown` (30). Imputed invalid/missing ages using city-level medians (0 missing remain).
  2. *Orders*: Deduplicated identical rows and kept latest timestamp per `order_id`. Imputed missing amounts via product catalogue prices (`price * qty * (1 - disc)`). Set negative delivery minutes to missing; imputed missing completed delivery times using category medians; left cancelled orders as missing (1,199 rows).
  3. *Validation*: Exactly 11,110 completed orders totaling **INR 2,998,668.80** stored in `data/instamart.db`.

### Analytical SQL Highlights (`sql/queries.sql`)
- **Q1 (High-Value Filter)**: 30 customers with orders > 500 totaling > 8,000.
- **Q2 (City-Category Breakdown)**: 30 cells totaling INR 2,998,668.80 (peak: Bengaluru / Electronics at INR 233,909.45).
- **Q3 (Above Average Price)**: 11 products priced above average.
- **Q4 (Zero Spend)**: 25 customers with zero completed orders.
- **Q5 (CTE Above Average Spend)**: 550 customers above mean spend (INR 2,552.06).
- **Q6 (Ranking Window Functions)**: Top spender per city (Mumbai lead: C0693 at INR 14,766.55). Exactly 8 rows show `RANK() != ROW_NUMBER()` due to customer spend ties.
- **Q7 (Running Total)**: 183 daily rows; cumulative completed revenue reaches **INR 2,998,668.80** on day 183 (`SUM(revenue) OVER (ORDER BY day)`).

---

## Part B

### RFM & K-Means Clustering ($K=4$)
- **Cohort**: 1,175 active customers. Recency median: 19.0d (0–180d); Frequency mean: 9.46 (max 33); Monetary mean: INR 2,552.06 (max INR 14,766.55).
- **Elbow & Silhouette**: Evaluated $K=2..8$. Peak silhouette achieved at **$K=4$ (0.526, Inertia: 743.1)**.
- **Segment Profiles**:
  - *High-Value Spenders (Premium)*: Size 138 | Mean Monetary: INR 7,129.70 | Frequency: 6.8
  - *Loyal Champions*: Size 419 | Mean Recency: 8.8d | Frequency: 19.4 | Monetary: INR 3,410.02
  - *Occasional Shoppers*: Size 436 | Mean Recency: 28.9d | Frequency: 3.8 | Monetary: INR 1,057.62
  - *Dormant / At-Risk*: Size 182 | Mean Recency: 109.0d | Frequency: 2.2 | Monetary: INR 685.99
- **PCA Dimensionality Reduction**: 2D projection captures **85.2%** of total variance (PC1: 0.634, PC2: 0.218). Exported to `outputs/customer_segments.csv`.

### Lapse Prediction (Cutoff: 2025-09-01)
- **Cohort**: 1,156 customers with pre-cutoff completed orders; 443 lapsed (38.32%). Stratified 80/20 train/test split (924 train, 232 test).
- **No Data Leakage**: Scaler and OneHotEncoder fitted strictly on training data (Learned frequency mean: 8.11 vs cohort mean: 8.03). Zero features derived from post-cutoff data or Task 3 segment labels.
- **Baseline**: Always predicting active yields 61.6% accuracy and **0.000** lapsed F1.
- **Model Evaluation**:
  - *Logistic Regression (GridSearchCV C=100)*: Test F1 = **0.648** (Precision: 0.644, Recall: 0.652).
  - *Random Forest (GridSearchCV depth=8, leaf=10)*: Test F1 = **0.640** (Precision: 0.640, Recall: 0.640).
- **Class-Balance Experiment**: Oversampling train set to 570 per class boosted Logistic Regression recall from 0.652 to **0.854** (precision: 0.589).
- **Key Features**: Monetary (0.278), Frequency (0.277), Recency (0.111), Delivery Minutes (0.062).
- **Artifacts**: Saved `outputs/lapse_model.joblib` and `outputs/retention_list.csv` (10 high-risk premium customers).

---

## Part C

### Privacy Guardrails & Triage Pipeline
- **Input Screening**: 164 total tickets; 4 skipped empty, 4 flagged long, 156 passed for processing.
- **PII Masking**: Successfully redacted 55 email addresses, 91 phone numbers, and 132 order IDs using ordered regex rules.
- **Resilience**: 19 transient 503 errors caught and automatically resolved via exponential backoff retries.

### Evaluation & Segment-Based Routing
- **Output Validation**: 135 valid JSON parses, 12 keyword fallback triggers, 9 flagged invalid outputs.
- **Prompt Performance**: Zero-shot accuracy = **85.7%**; Few-shot accuracy = **100.0%**.
- **Token Budget Demonstration**: Run with 6,000 token limit safely stopped after budget exhaustion, routing 58 tickets to `human_review` in `outputs/triage_budget_demo.csv`.
- **Final Routing (`outputs/triage_results.csv`)**: Integrated Part B segment data to route fraud tickets to `security_team` (HIGH priority), premium customer issues to `premium_desk` (HIGH priority), and standard tickets to `support_queue`.

---

## Part D

### Policy RAG Assistant
- **Corpus**: 5 policy documents split into 15 chunks (40 words with 10-word overlap).
- **Retrieval & Accuracy**: 12/12 (100%) on answerable queries using TF-IDF cosine similarity; 2/2 unanswerable queries correctly triggered the safe fallback disclaimer.
- **Chunking Benchmark**: Overlap chunking preserved multi-sentence context boundaries, outperforming zero-overlap chunking on cross-sentence rules.

### Autonomous Refund Agent ReAct Loop
Tested across all 7 benchmark scenarios from `data/raw/agent_scenarios.csv`:

| Scenario | User Intent | Agent Outcome | Refund (INR) | Model Calls | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | Late delivery (>45m) small order | REFUND | ₹128.25 | 5 | Success |
| **S2** | Late delivery (>45m) small order | REFUND | ₹135.00 | 5 | Success |
| **S3** | Late delivery (>45m) high value | ESCALATE | ₹1,999.00 | 5 | Human Lead Escalate |
| **S4** | On-time delivery refund request | DENIED | ₹0.00 | 4 | Denied (Within SLA) |
| **S5** | Cancelled order refund request | DENIED | ₹0.00 | 3 | Denied (Not Charged) |
| **S6** | Non-existent order ID | NOT_FOUND | ₹0.00 | 2 | Not Found |
| **S7** | Simulated cyclic policy loop | FALLBACK | ₹0.00 | 6 | `FALLBACK_MAX_ITERATIONS` |

### Safety Boundaries & State Persistence Demonstrations
- **Token Limit Stop**: S1 run with `token_limit=200` halted before executing tool calls with status `FALLBACK_TOKEN_LIMIT`.
- **Timeout Stop**: S1 run with `time_limit_seconds=0` halted before first model call with status `FALLBACK_TIMEOUT`.
- **Crash & Resume Demonstration**: S1 crashed deliberately after step 2 checkpoint (`search_policy`). Resumed via `resume_agent(run_id)` without duplicating steps. Executed tools in exact order: `get_order` → `search_policy` → `[CRASH & RESUME]` → `calculate_refund` → `issue_refund` (Final status: REFUND ₹128.25).
- **Logs**: Saved to `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, and `state/`.

---

## Limitations and Next Steps

1. **Synthetic Data Boundaries**: Orders and ticket logs are synthetically generated and do not model complex multi-item split deliveries or extreme weather route disruptions.
2. **Post-Hoc Retention Flag**: The `is_premium` label in `outputs/retention_list.csv` is derived from full-period RFM metrics (including the September evaluation window); it serves as an operational campaign list rather than a strictly leakage-free evaluation artifact.
3. **Sparse vs. Dense Retrieval**: The offline TF-IDF retriever relies on keyword lexical matching. Integrating dense vector embeddings (e.g., local sentence-transformers) would improve semantic understanding of colloquial refund queries.
4. **Tool Set Expansion**: Extending the refund agent to support payment gateway reconciliation, customer address verification, and delivery photo inspection.

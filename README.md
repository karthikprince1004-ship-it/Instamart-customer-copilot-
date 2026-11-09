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

### Pandas Cross-Check & Exploratory Analysis
- **Q2 Pivot Cross-Check**: `pd.pivot_table(..., margins=True)` grand total equals exactly **INR 2,998,668.80**, matching the SQL total across all 30 city-category cells.
- **City Aggregations (`groupby` completed orders)**:
  - *Bengaluru*: Sum INR 523,643.40 | Mean INR 276.47 | Count 1,894
  - *Mumbai*: Sum INR 479,058.50 | Mean INR 297.00 | Count 1,613
  - *Delhi-NCR*: Sum INR 508,412.30 | Mean INR 265.91 | Count 1,912
- **Distribution & Outlier Fences (`order_amount`)**:
  - Mean: INR 269.91 vs. Median: INR 135.00 (heavily right-skewed, skewness = 4.61).
  - Q1: INR 64.80 | Q3: INR 297.00 | IQR: INR 232.20.
  - Fences: Lower = -INR 283.50 (0 outliers below) | Upper = INR 645.30 (897 outliers above).

> **Exploratory Data Analysis Summary (Under 200 words):**  
> This dataset records quick-commerce grocery and household transactions across six Indian metropolitan markets. Order spend exhibits a pronounced right-skewed distribution where the median purchase is modest (INR 135.00) while bulk grocery runs and electronics purchases stretch the mean to INR 269.91 with skewness of 4.61. The 1.5×IQR upper fence identifies 897 high-value outliers (above INR 645.30), representing legitimate high-ticket orders rather than data entry anomalies. Delivery times center around 30 minutes with completed orders capped under SLA thresholds. Across completed transactions, the correlation matrix reveals that order amount has a slight negative correlation with discount percentage (r ≈ -0.095) and a slight positive correlation with item quantity (r ≈ +0.087), while delivery minutes remain virtually uncorrelated with basket size (r ≈ -0.012). The strongest linear association exists between order amount and discount percentage (-0.095). However, a correlation of this modest magnitude does not establish causation: promotional discounts may be selectively targeted at lower-priced staple items rather than directly causing smaller checkout totals.

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

### 1. Input Guardrails & PII Redaction
- **Screening & Guardrails (Task 1)**: Evaluated all 164 support tickets.
  - **Empty tickets (`skipped_empty`)**: **4 tickets** (`T0157`, `T0158`, `T0159`, `T0160`) contain no text/whitespace and were halted before model invocation (0 tokens used).
  - **Long tickets (`flagged_long`)**: **4 tickets** (`T0161`, `T0162`, `T0163`, `T0164`) each estimate to **734 tokens** (exceeding `MAX_INPUT_TOKENS = 400`) and were halted before model invocation (0 tokens used).
  - **Eligible tickets**: **156 tickets** sent to triage pipeline.
- **PII Redaction (Task 2)**: Applied ordered regex transformations (`[EMAIL]` -> `[PHONE]` -> `[ORDER_ID]`).
  - Redacted **55 email addresses**, **91 phone numbers**, and **132 order IDs** (**278 substitutions total**).
  - Re-applying all three regex patterns to redacted texts confirmed **0 remnants left**.

  ### 2. Zero-Shot vs. Few-Shot Prompt Design (Task 3)
- **Zero-Shot Prompt**: Contains the directive `CLASSIFY`, specifies the four categories (`billing`, `delivery`, `technical`, `fraud`), mandates strict JSON output `{"category": "...", "reason": "..."}`, and contains **no** `EXAMPLES:` marker.
- **Few-Shot Prompt**: Shares the same system instructions but introduces an explicit `EXAMPLES:` block containing two labeled exemplars (including an unauthorized card transaction demonstrating the `fraud` category).
- **Behavioral Impact**: The few-shot prompt prioritizes `fraud` resolution over generic billing keywords during boundary tie-breaks, accurately escalating security tickets.


### 2. Resilient Execution & Response Validation
- **Transient Error Resilience (Task 4)**: Across 156 model calls, caught exactly **19 TransientLLMError** (503 server busy) exceptions. All 19 were recovered on retry attempt 0 with exponential backoff and jitter (delays between 0.0450s and 0.0575s). Zero permanent failures.
- **Output Validation (Task 5)**:
  - `classified_json`: **135 tickets** parsed as valid JSON with category in allowed list.
  - `classified_keyword`: **12 tickets** parsed via regex lowercase word extraction (`re.findall(r"[a-z]+", text.lower())`).
  - `flagged_invalid_output`: **9 tickets** routed safely to `human_review`.
  - Total classified: **147 tickets** (135 JSON + 12 keyword).

### 3. Zero-Shot vs. Few-Shot Comparison (Task 6)
Both prompts enforce the `CLASSIFY` directive and require strict JSON output.

| Metric | Zero-Shot Run | Few-Shot Run | Reference |
| :--- | :--- | :--- | :--- |
| **Tickets Classified** | 147 / 156 | 147 / 156 | 147 |
| **Accuracy** | **85.7%** (126 / 147) | **100.0%** (147 / 147) | 85.7% vs 100.0% |
| **Total Tokens** | 10,180 | 25,312 | - |
| **Tokens / Classified Ticket** | 69.3 | 172.2 | - |

#### Zero-Shot Confusion Matrix
| Actual \ Predicted | billing | delivery | fraud | technical |
| :--- | :---: | :---: | :---: | :---: |
| **billing** | 38 | 0 | 0 | 0 |
| **delivery** | 0 | 36 | 0 | 0 |
| **fraud** | **11** | 0 | 23 | 0 |
| **technical** | **10** | 0 | 0 | 29 |

> **Cost vs. Quality Tradeoff:**  
> The few-shot prompt consumes ~2.5× more tokens per ticket (172.2 vs. 69.3 tokens), adding roughly 103 tokens per ticket. However, zero-shot misclassifies 11 fraud tickets and 10 technical tickets as billing because payment-related vocabulary in customer complaints overrides subtle account takeover or app-crash cues. In quick commerce, routing account compromise tickets to a standard billing queue severely delays fraud containment and increases financial risk. The 14.3 percentage-point accuracy gain (85.7% to 100.0%) and complete elimination of fraud leakage overwhelmingly justify the additional token cost.

### 4. Token Budget & Segment-Based Routing (Task 7)
- **Token Budget Demonstration (`outputs/triage_budget_demo.csv`)**: Evaluated zero-shot pipeline with `TOKEN_BUDGET = 6,000`. Execution safely halted when cumulative usage reached the ceiling, marking **64 tickets** as `budget_exhausted` and routing them to `human_review`.
- **Production Triage Results (`outputs/triage_results.csv`)**: Generated from few-shot pipeline (`TOKEN_BUDGET = 100,000`) across all **164 rows** using Part B customer segments:
  - `support_queue` (NORMAL priority): **102 tickets** (standard inquiries)
  - `security_team` (HIGH priority): **34 tickets** (all fraud issues)
  - `human_review` (NORMAL priority): **17 tickets** (4 empty, 4 long, 9 invalid output)
  - `premium_desk` (HIGH priority): **11 tickets** (non-fraud issues from `is_premium = True` customers)

---

## Part D

### 1. Document Chunking & TF-IDF Retrieval (Tasks 1–3)
- **Corpus Chunking**: 5 policy text files split into **15 chunks** (40 words with 10-word overlap):
  - `refund_policy.txt`: 4 chunks
  - `delivery_policy.txt`: 3 chunks
  - `escalation_policy.txt`: 3 chunks
  - `payment_policy.txt`: 3 chunks
  - `privacy_policy.txt`: 2 chunks
- **TF-IDF & Cosine Similarity**: Term Frequency (TF) measures how often a word appears in a chunk relative to chunk length, while Inverse Document Frequency (IDF) penalizes words common across all documents to amplify discriminative terms. A fundamental limitation is lexical vocabulary mismatch; queries with identical intent but alternative wording (e.g., "reimbursement" vs. "refund") score zero similarity without dense semantic embeddings.
- **Top-3 Retrieval Check**: Query `"refund policy late delivery"` returns all top-3 chunks from `refund_policy.txt` (Reference matched).

### 2. Retrieval & Grounding Evaluation Benchmark (Task 4)
Evaluated across 14 benchmark questions from `data/raw/policy_eval.csv` (12 answerable, 2 out-of-scope):

| Metric | 10-Word Overlap (Production) | 0-Word Overlap (Zero Overlap) |
| :--- | :---: | :---: |
| **Total Chunks** | 15 chunks | 13 chunks |
| **Document Recall (Expected Doc in Top-3)** | **12 / 12** (100.0%) | **12 / 12** (100.0%) |
| **Grounded Fact Accuracy** | **12 / 12** (100.0%) | **10 / 12** (83.3%) |
| **Unanswerable Safe Fallback Rate** | **2 / 2** (100.0%) | **2 / 2** (100.0%) |

> **Chunk Boundary Failure (Zero Overlap):**  
> Under zero overlap, the question *"What delivery time makes a completed order eligible for a full refund?"* failed to extract the expected fact `"more than 45 minutes"`. Without overlap, the conditional phrase was severed across chunk boundaries, yielding only the trailing clause (`"receives a full refund of the order amount."`). The 10-word overlap preserved context across sentence boundaries and achieved 100% grounded accuracy.

### 3. Autonomous Refund Agent ReAct Benchmark (Task 6)
Evaluated across all 7 benchmark scenarios from `data/raw/agent_scenarios.csv`:

| Scenario ID | Status | Refund Amount (INR) | Model Calls | Tokens Used |
| :---: | :---: | :---: | :---: | :---: |
| **S1** | REFUND | ₹128.25 | 5 | 927 |
| **S2** | REFUND | ₹135.00 | 5 | 927 |
| **S3** | ESCALATE | ₹1,999.00 | 5 | 914 |
| **S4** | DENIED | ₹0.00 | 4 | 629 |
| **S5** | DENIED | ₹0.00 | 3 | 355 |
| **S6** | NOT_FOUND | ₹0.00 | 2 | 84 |
| **S7** | FALLBACK_MAX_ITERATIONS | ₹0.00 | 6 | 3,036 |

*S1 took exactly 5 model calls and refunded ₹128.25. Under the default `token_limit=6000`, S7 completed all 6 model calls without premature termination.*

### 4. Safety Stops & Resumable Checkpoints (Tasks 7 & 8)
- **Token Limit Stop**: S1 evaluated with `token_limit=200` halted before executing tool calls with status `FALLBACK_TOKEN_LIMIT`.
- **Timeout Stop**: S1 evaluated with `time_limit_seconds=0` halted before the first model call with status `FALLBACK_TIMEOUT`.
- **Crash & Resume Demonstration**: S1 crashed deliberately after step 2 (`search_policy`) and was resumed via `resume_agent(run_id)`:
  - **Tool execution sequence**: `get_order` → `search_policy` → `[CRASH & RESUME]` → `calculate_refund` → `issue_refund` (4 executions total, none repeated).
  - **Final status**: `REFUND` with ₹128.25.
- **Persistent Logs**: Stored in `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, and `state/`.

---

### 5. ReAct vs. Plan-and-Execute Architecture (Task 9)
> **Comparative Analysis (Under 250 words):**  
> In our quick-commerce refund workflow, the **ReAct (Reason + Act)** pattern interleaves model reasoning with immediate tool observations, while a **Plan-and-Execute** architecture generates an entire multi-step plan up front and executes tools sequentially, replanning only upon failure.
> 
> Across our benchmark runs, ReAct required dynamic call counts scaling with intent ambiguity: straightforward cases exited early (2 calls for S6 non-existent orders; 3 calls for S5 cancelled orders), while full late-delivery validations required 5 calls (S1, S2, S4) and cyclic edge cases halted at 6 iterations. A plan-and-execute model uses fewer upfront planning calls (1 plan generation + 1 final synthesis = ~2 calls) and fewer cumulative tokens for ideal paths. 
> 
> However, plan-and-execute degrades significantly when encountering runtime branch surprises. In S6 (`NOT_FOUND`), a pre-planned schedule (`get_order` → `search_policy` → `calculate_refund` → `issue_refund`) attempts to execute downstream tools on null data or triggers an expensive replanning cycle. In contrast, ReAct immediately inspects the `{"error": "Order not found"}` observation from `get_order` and halts dynamically in 2 calls without calling refund calculations. 
> 
> Furthermore, under infinite loops (S7), plan-and-execute can loop during replanning cycles if error feedback is ambiguous, whereas our ReAct implementation enforces deterministic iteration and token tripwires that safely terminate cyclic prompts without human intervention.

## Limitations and Next Steps

- **Synthetic Distribution Constraints**: The dataset simulates quick-commerce behavior with a fixed 6-month window (13,084 raw orders across 1,200 users) and does not capture live variables such as dynamic traffic congestion, rain surges, or multi-store stockouts.
- **Lexical Retrieval Limitations**: The Policy RAG assistant relies on TF-IDF cosine similarity, which works well for exact keyword matching but cannot resolve semantic synonyms (e.g., "reimbursement" vs. "refund") without dense vector embeddings.
- **Single-Turn Triage & Deterministic Mocking**: Ticket classification and agent tools evaluate isolated customer turns using a deterministic mock LLM; multi-turn dialog history and edge-case negotiation are not yet modeled.
- **Next Steps & Production Scaling**: Integrate dense vector retrieval (e.g., OpenAI/sentence-transformers embeddings), deploy real-time webhook listeners for order events, add multi-turn conversation memory, and link automated ledger writes to core banking/payment gateways.

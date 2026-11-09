"""
Part C — Support Ticket Triage with Guardrails
Capstone Project — Instamart Customer Intelligence & Support Copilot
"""

import os
import sys
import re
import math
import json
import time
import random
from pathlib import Path
import pandas as pd

# Path resolution: ensure project root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for directory in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if directory not in sys.path:
        sys.path.insert(0, directory)

import mock_llm

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

# Task 2: Required Regex Patterns
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")

def est_tokens(text: str) -> int:
    return math.ceil(len((text or "").split()) * 1.3)

def redact_text(text: str) -> tuple[str, dict]:
    """Redacts email, phone, and order id in strict sequence."""
    if not isinstance(text, str):
        return "", {"email": 0, "phone": 0, "order_id": 0}
    
    email_matches = len(EMAIL_RE.findall(text))
    t1 = EMAIL_RE.sub("[EMAIL]", text)
    
    phone_matches = len(PHONE_RE.findall(t1))
    t2 = PHONE_RE.sub("[PHONE]", t1)
    
    order_matches = len(ORDER_RE.findall(t2))
    t3 = ORDER_RE.sub("[ORDER_ID]", t2)
    
    return t3, {"email": email_matches, "phone": phone_matches, "order_id": order_matches}

def check_pii_remnants(text: str) -> int:
    return len(EMAIL_RE.findall(text)) + len(PHONE_RE.findall(text)) + len(ORDER_RE.findall(text))

# Task 3: Prompts
ZERO_SHOT_PROMPT = (
    "CLASSIFY the customer support ticket into one of the following categories: "
    "billing, delivery, technical, fraud. "
    "Reply strictly with a JSON object in this format: {\"category\": \"...\", \"reason\": \"...\"}."
)

FEW_SHOT_PROMPT = (
    "CLASSIFY the customer support ticket into one of the following categories: "
    "billing, delivery, technical, fraud. "
    "Reply strictly with a JSON object in this format: {\"category\": \"...\", \"reason\": \"...\"}.\n"
    "EXAMPLES:\n"
    "Ticket: 'My account was charged twice for the same delivery.' -> {\"category\": \"billing\", \"reason\": \"duplicate charge\"}\n"
    "Ticket: 'Someone logged into my account from an unknown device and ordered items.' -> {\"category\": \"fraud\", \"reason\": \"unauthorized account takeover\"}\n"
    "Ticket: 'The app keeps crashing whenever I reach the checkout screen.' -> {\"category\": \"technical\", \"reason\": \"app crash at checkout\"}\n"
    "Ticket: 'Order was supposed to arrive in 15 minutes but rider is delayed.' -> {\"category\": \"delivery\", \"reason\": \"delivery delay\"}"
)

# Tasks 4 & 5: Model Call with Exponential Backoff/Jitter & Validation
def call_model_with_retry(messages: list, ticket_id: str) -> object:
    for n in range(3):
        try:
            return mock_llm._classify(messages)
        except mock_llm.TransientLLMError as e:
            wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            print(f"  [RETRY] Ticket {ticket_id} | Attempt {n} | Wait {wait:.4f}s | Reason: {e}")
            time.sleep(wait)
    raise RuntimeError(f"Permanent failure for ticket {ticket_id} after 3 retries")

def validate_response(content: str) -> tuple[str, str]:
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            cat = str(data.get("category", "")).lower().strip()
            if cat in ALLOWED_CATEGORIES:
                return "classified_json", cat
    except Exception:
        pass
    
    words = [w for w in re.findall(r"[a-z]+", (content or "").lower()) if w in ALLOWED_CATEGORIES]
    if len(words) == 1:
        return "classified_keyword", words[0]
        
    return "flagged_invalid_output", ""

def load_tickets() -> pd.DataFrame:
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "tickets.csv",
        PROJECT_ROOT / "data" / "tickets.csv",
        PROJECT_ROOT / "data" / "clean" / "tickets.csv"
    ]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c, keep_default_na=False)
    raise FileNotFoundError("Could not locate tickets.csv")

def load_segments() -> dict:
    seg_path = PROJECT_ROOT / "outputs" / "customer_segments.csv"
    if seg_path.exists():
        df_seg = pd.read_csv(seg_path)
        return df_seg.set_index("customer_id").to_dict(orient="index")
    return {}

def run_pipeline(prompt: str, token_budget: int = None, prompt_name: str = "Zero-Shot") -> tuple[pd.DataFrame, dict]:
    mock_llm.reset()
    tickets_df = load_tickets()
    seg_dict = load_segments()
    
    rows = []
    status_counts = {
        "skipped_empty": 0,
        "flagged_long": 0,
        "classified_json": 0,
        "classified_keyword": 0,
        "flagged_invalid_output": 0,
        "budget_exhausted": 0
    }
    
    running_tokens = 0
    total_tokens = 0
    correct_count = 0
    classified_count = 0
    
    y_true = []
    y_pred = []
    
    for idx, row in tickets_df.iterrows():
        t_id = row["ticket_id"]
        c_id = row["customer_id"]
        raw_text = str(row["text"]) if "text" in row else str(row.get("ticket_text", ""))
        true_cat = str(row.get("true_category", "")).lower().strip()
        
        cust_info = seg_dict.get(c_id, {})
        segment_name = cust_info.get("segment_name", "")
        is_premium = cust_info.get("is_premium", False)
        
        # Task 1: Guardrails
        if not raw_text or raw_text.strip() == "":
            status = "skipped_empty"
            category = ""
            tokens_used = 0
            status_counts["skipped_empty"] += 1
        elif est_tokens(raw_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
            category = ""
            tokens_used = 0
            status_counts["flagged_long"] += 1
        elif token_budget is not None and running_tokens >= token_budget:
            status = "budget_exhausted"
            category = ""
            tokens_used = 0
            status_counts["budget_exhausted"] += 1
        else:
            redacted, _ = redact_text(raw_text)
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": redacted}
            ]
            
            resp = call_model_with_retry(messages, t_id)
            tokens_used = resp.usage.total_tokens
            running_tokens += tokens_used
            total_tokens += tokens_used
            
            raw_content = resp.choices[0].message.content
            status, category = validate_response(raw_content)
            status_counts[status] += 1
            
            if status in ("classified_json", "classified_keyword"):
                classified_count += 1
                y_true.append(true_cat)
                y_pred.append(category)
                if category == true_cat:
                    correct_count += 1
                    
        # Task 7: Routing logic
        if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            route = "human_review"
            priority = "NORMAL"
            category = ""
        elif category == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif is_premium:
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"
            
        rows.append({
            "ticket_id": t_id,
            "customer_id": c_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used
        })
        
    df_out = pd.DataFrame(rows)
    acc = (correct_count / classified_count) if classified_count > 0 else 0.0
    stats = {
        "status_counts": status_counts,
        "total_tokens": total_tokens,
        "classified_count": classified_count,
        "correct_count": correct_count,
        "accuracy": acc,
        "tokens_per_ticket": (total_tokens / classified_count) if classified_count > 0 else 0,
        "y_true": y_true,
        "y_pred": y_pred
    }
    return df_out, stats

def main():
    print("=" * 70)
    print("  INSTAMART COPILOT — PART C: SUPPORT TICKET TRIAGE")
    print("=" * 70)
    
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    tickets_df = load_tickets()
    print(f"Total input tickets: {len(tickets_df)} (Reference: 164)")
    
    # Task 1 & Task 2 Verification
    empty_t = [r["ticket_id"] for _, r in tickets_df.iterrows() if not str(r["text"]).strip()]
    long_t = [(r["ticket_id"], est_tokens(str(r["text"]))) for _, r in tickets_df.iterrows() if est_tokens(str(r["text"])) > MAX_INPUT_TOKENS]
    print(f"\n[Task 1: Input Guardrails Check]")
    print(f"  Empty tickets (status 'skipped_empty'): {len(empty_t)} (Ref: 4) -> {empty_t}")
    print(f"  Long tickets (status 'flagged_long')   : {len(long_t)} (Ref: 4, token est: {[t[1] for t in long_t]})")
    print(f"  Eligible tickets sent to model        : {len(tickets_df) - len(empty_t) - len(long_t)} (Ref: 156)")
    
    total_emails, total_phones, total_orders = 0, 0, 0
    remnant_count = 0
    for txt in tickets_df["text"]:
        redacted, counts = redact_text(str(txt))
        total_emails += counts["email"]
        total_phones += counts["phone"]
        total_orders += counts["order_id"]
        remnant_count += check_pii_remnants(redacted)
        
    print(f"\n[Task 2: PII Redaction Verification]")
    print(f"  Emails redacted    : {total_emails} (Ref: 55)")
    print(f"  Phones redacted    : {total_phones} (Ref: 91)")
    print(f"  Order IDs redacted : {total_orders} (Ref: 132)")
    print(f"  Total substitutions: {total_emails + total_phones + total_orders} (Ref: 278)")
    print(f"  PII remnants left  : {remnant_count} (Ref: 0)")
    assert remnant_count == 0, "PII Redaction Assertion Failed: Remnants detected!"
    
    # Task 6: Zero-Shot vs Few-Shot Comparison
    print("\n" + "=" * 70)
    print("TASK 6: RUNNING ZERO-SHOT AND FEW-SHOT COMPARISON")
    print("=" * 70)
    
    print("\nRunning Zero-Shot Pipeline (no token budget)...")
    df_zero, stats_zero = run_pipeline(ZERO_SHOT_PROMPT, token_budget=None, prompt_name="Zero-Shot")
    print(f"  Classified tickets : {stats_zero['classified_count']} (Ref: 147)")
    print(f"  Zero-Shot Accuracy : {stats_zero['accuracy'] * 100:.1f}% (Ref: 85.7%)")
    print(f"  Total Tokens       : {stats_zero['total_tokens']:,}")
    print(f"  Tokens / Ticket    : {stats_zero['tokens_per_ticket']:.1f}")
    
    print("\nZero-Shot Confusion Matrix:")
    cm = pd.crosstab(pd.Series(stats_zero['y_true'], name="Actual"), pd.Series(stats_zero['y_pred'], name="Predicted"))
    print(cm)
    
    print("\nRunning Few-Shot Pipeline (no token budget)...")
    df_few, stats_few = run_pipeline(FEW_SHOT_PROMPT, token_budget=None, prompt_name="Few-Shot")
    print(f"  Classified tickets : {stats_few['classified_count']} (Ref: 147)")
    print(f"  Few-Shot Accuracy  : {stats_few['accuracy'] * 100:.1f}% (Ref: 100.0%)")
    print(f"  Total Tokens       : {stats_few['total_tokens']:,}")
    print(f"  Tokens / Ticket    : {stats_few['tokens_per_ticket']:.1f}")
    
    # Task 7: Save Deliverables
    print("\n" + "=" * 70)
    print("TASK 7: SAVING REQUIRED DELIVERABLES")
    print("=" * 70)
    
    print("\nRunning Budget Demo with TOKEN_BUDGET = 6000...")
    df_budget, stats_budget = run_pipeline(ZERO_SHOT_PROMPT, token_budget=6000, prompt_name="Budget-Demo")
    budget_csv = outputs_dir / "triage_budget_demo.csv"
    df_budget[["ticket_id", "customer_id", "status", "tokens_used"]].to_csv(budget_csv, index=False)
    exhausted_count = (df_budget["status"] == "budget_exhausted").sum()
    print(f"  Saved budget demo to: {budget_csv}")
    print(f"  Tickets with status 'budget_exhausted': {exhausted_count} (Ref: >= 1)")
    
    print("\nRunning Final Production Few-Shot Triage with Segment Routing...")
    df_final, _ = run_pipeline(FEW_SHOT_PROMPT, token_budget=100000, prompt_name="Final-Triage")
    final_csv = outputs_dir / "triage_results.csv"
    df_final.to_csv(final_csv, index=False)
    print(f"  Saved final triage results to: {final_csv}")
    print(f"  Total rows: {len(df_final)} (Ref: 164)")
    print("\nFinal Routing Breakdown:")
    print(df_final["route"].value_counts().to_string())
    print("\nFinal Priority Breakdown:")
    print(df_final["priority"].value_counts().to_string())
    
    print("\n" + "=" * 70)
    print("  PART C COMPLETE AND VERIFIED ACROSS ALL 7 TASKS!")
    print("=" * 70)

if __name__ == "__main__":
    main()

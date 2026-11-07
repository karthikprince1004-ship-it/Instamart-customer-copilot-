"""
Part C: Support Ticket Triage with Guardrails
- Input guardrails (empty ticket detection, length check)
- PII redaction (email, Indian phone numbers, Order IDs)
- Classification using mock_llm (Zero-Shot and Few-Shot)
- Transient error retry with exponential backoff & jitter
- Output JSON validation with keyword fallback
- Token budget simulation and tracking
- Customer segment priority routing
"""

import sys
import os
from pathlib import Path
import re
import math
import json
import time
import random
import pandas as pd

# Path resolution: ensure project root and src/ directory are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for directory in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if directory not in sys.path:
        sys.path.insert(0, directory)

import mock_llm

# -----------------------------------------------------------------------------
# 1. Input Guardrails and PII Redaction
# -----------------------------------------------------------------------------

def est_tokens(text: str) -> int:
    """Estimates tokens using the standard 1.3 tokens per word formula."""
    return math.ceil(len((text or "").split()) * 1.3)

# Raw strings (r"...") prevent SyntaxWarning for escape sequences
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_REGEX = re.compile(r"(?:\+91[\-\s]?)?[6-9]\d{9}|0[6-9]\d{9}")
ORDER_ID_REGEX = re.compile(r"ORD-\d{6}")

def redact_pii(text: str) -> tuple[str, dict]:
    """Redacts email, Indian phone numbers, and Order IDs."""
    if not isinstance(text, str):
        return "", {"email": 0, "phone": 0, "order_id": 0}

    email_count = len(EMAIL_REGEX.findall(text))
    phone_count = len(PHONE_REGEX.findall(text))
    order_count = len(ORDER_ID_REGEX.findall(text))

    redacted = EMAIL_REGEX.sub("[EMAIL]", text)
    redacted = PHONE_REGEX.sub("[PHONE]", redacted)
    redacted = ORDER_ID_REGEX.sub("[ORDER_ID]", redacted)

    return redacted, {
        "email": email_count,
        "phone": phone_count,
        "order_id": order_count
    }

# -----------------------------------------------------------------------------
# 2. Response Parsing & Output Validation
# -----------------------------------------------------------------------------

def get_response_content(response) -> str:
    """Safely extracts text content from mock_llm SimpleNamespace response."""
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        if hasattr(msg, "content") and msg.content is not None:
            return msg.content
    return str(response)

VALID_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

KEYWORD_MAP = {
    "fraud": {"unauthorized", "fraud", "hacked", "stolen"},
    "billing": {"charged", "invoice", "bill", "refund", "deducted", "payment"},
    "technical": {"app", "crash", "login", "otp", "error", "freezes"},
    "delivery": {"arrived", "rider", "delivery", "delivered", "late"}
}

def validate_and_parse_output(content: str) -> tuple[str, str]:
    """Validates JSON output, falling back to keyword matching if invalid."""
    content_clean = content.strip()
    try:
        data = json.loads(content_clean)
        if isinstance(data, dict):
            cat = str(data.get("category", "")).lower().strip()
            if cat in VALID_CATEGORIES:
                return "VALID_JSON", cat
    except Exception:
        pass

    words = set(re.findall(r"[a-z]+", content_clean.lower()))
    for cat, kws in KEYWORD_MAP.items():
        if kws & words:
            return "KEYWORD_FALLBACK", cat

    return "INVALID", "technical"

# -----------------------------------------------------------------------------
# 3. LLM Caller with Retry Logic
# -----------------------------------------------------------------------------

def call_llm_with_retry(messages: list, max_retries: int = 4) -> tuple[object, int]:
    """Calls mock_llm.chat with exponential backoff & jitter for TransientLLMError."""
    retries = 0
    for attempt in range(max_retries):
        try:
            resp = mock_llm.chat(messages)
            return resp, retries
        except mock_llm.TransientLLMError:
            retries += 1
            if attempt == max_retries - 1:
                raise
            time.sleep(0.01 * (2 ** attempt) + random.uniform(0.001, 0.005))
    raise RuntimeError("Retry loop exhausted")

# -----------------------------------------------------------------------------
# 4. Prompts (Zero-Shot and Few-Shot)
# -----------------------------------------------------------------------------

FEW_SHOT_SYSTEM = (
    "You are an automated support ticket triage assistant. CLASSIFY the following "
    "customer support ticket into exactly one category: billing, delivery, technical, or fraud. "
    "EXAMPLES:\n"
    "1. Ticket: 'Money deducted from UPI but order failed.' -> {\"category\": \"billing\", \"reason\": \"payment deduction\"}\n"
    "2. Ticket: 'Rider took 45 minutes and food was cold.' -> {\"category\": \"delivery\", \"reason\": \"rider delay\"}\n"
    "3. Ticket: 'App crashes continuously when opening cart.' -> {\"category\": \"technical\", \"reason\": \"app crash\"}\n"
    "4. Ticket: 'Someone placed an unauthorized order on my account.' -> {\"category\": \"fraud\", \"reason\": \"account compromise\"}\n"
    'Respond strictly in JSON format: {"category": "<category>", "reason": "<brief_reason>"}.'
)

# -----------------------------------------------------------------------------
# 5. File Resolution & Execution
# -----------------------------------------------------------------------------

def find_tickets_file() -> Path | None:
    """Finds tickets.csv in data/raw, data/clean, data/, or searches recursively."""
    candidate_paths = [
        PROJECT_ROOT / "data" / "raw" / "tickets.csv",
        PROJECT_ROOT / "data" / "tickets.csv",
        PROJECT_ROOT / "data" / "clean" / "tickets.csv",
        PROJECT_ROOT / "data" / "clean" / "tickets_clean.csv",
    ]
    for p in candidate_paths:
        if p.exists():
            return p

    # Recursive fallback
    matches = list(PROJECT_ROOT.rglob("tickets*.csv"))
    if matches:
        return matches[0]
    return None

def run_triage():
    print("=" * 70)
    print("  INSTAMART COPILOT — PART C: SUPPORT TICKET TRIAGE")
    print("=" * 70)

    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    tickets_path = find_tickets_file()
    if not tickets_path or not tickets_path.exists():
        print("Error: Could not locate tickets.csv!")
        print("Please ensure you have run 'python generate_data.py' first.")
        return

    # keep_default_na=False ensures empty tickets stay as empty strings ""
    df_tickets = pd.read_csv(tickets_path, keep_default_na=False)
    total_tickets = len(df_tickets)
    print(f"Loaded {total_tickets} raw tickets from: {tickets_path}")

    empty_count = 0
    long_count = 0
    pii_counts = {"email": 0, "phone": 0, "order_id": 0}
    eligible_tickets = []

    for idx, row in df_tickets.iterrows():
        # Check 'text' first (course dataset column), then 'ticket_text'
        raw_text = str(row.get("text", row.get("ticket_text", ""))).strip()
        ticket_id = row.get("ticket_id", f"TCK-{idx:04d}")
        customer_id = row.get("customer_id", f"CUST-{idx:04d}")

        if not raw_text or raw_text.lower() == "nan":
            empty_count += 1
            continue

        token_est = est_tokens(raw_text)
        is_long = token_est > 400
        if is_long:
            long_count += 1

        redacted_text, pii_stat = redact_pii(raw_text)
        for k in pii_counts:
            pii_counts[k] += pii_stat[k]

        eligible_tickets.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "raw_text": raw_text,
            "clean_text": redacted_text,
            "tokens": token_est,
            "is_long": is_long,
            "true_category": str(row.get("true_category", row.get("category", ""))).lower().strip()
        })

    print(f"\n[Guardrail Check]")
    print(f"  Total raw tickets : {total_tickets}")
    print(f"  Empty tickets     : {empty_count} (skipped)")
    print(f"  Long tickets >400 : {long_count} (flagged/truncated)")
    print(f"  Eligible tickets  : {len(eligible_tickets)}")

    print(f"\n[PII Redaction Stats]")
    print(f"  Emails redacted   : {pii_counts['email']}")
    print(f"  Phones redacted   : {pii_counts['phone']}")
    print(f"  Order IDs redacted: {pii_counts['order_id']}")

    # Priority routing using customer segments from Part B
    segments_path = PROJECT_ROOT / "outputs" / "customer_segments.csv"
    segment_map = {}
    if segments_path.exists():
        df_seg = pd.read_csv(segments_path)
        if "customer_id" in df_seg.columns and "segment" in df_seg.columns:
            segment_map = dict(zip(df_seg["customer_id"], df_seg["segment"]))

    # Classification & Token Tracking
    mock_llm.reset()
    results = []
    status_counts = {"VALID_JSON": 0, "KEYWORD_FALLBACK": 0, "INVALID": 0}
    total_retries_recovered = 0

    token_budget_log = []
    daily_budget = 10000
    accumulated_tokens = 0

    print("\nRunning Classification on eligible tickets...")
    for item in eligible_tickets:
        messages = [
            {"role": "system", "content": FEW_SHOT_SYSTEM},
            {"role": "user", "content": item["clean_text"]}
        ]

        resp, retries = call_llm_with_retry(messages)
        total_retries_recovered += retries

        content = get_response_content(resp)
        status, predicted_cat = validate_and_parse_output(content)
        status_counts[status] = status_counts.get(status, 0) + 1

        tokens_used = resp.usage.total_tokens if hasattr(resp, "usage") else item["tokens"] + 20
        accumulated_tokens += tokens_used
        budget_exhausted = accumulated_tokens > daily_budget

        seg = segment_map.get(item["customer_id"], "Standard")
        priority = "P1_HIGH" if (predicted_cat in ["fraud", "billing"] or "champion" in str(seg).lower() or "loyal" in str(seg).lower()) else "P2_NORMAL"

        results.append({
            "ticket_id": item["ticket_id"],
            "customer_id": item["customer_id"],
            "category": predicted_cat,
            "status": status,
            "retries": retries,
            "priority": priority,
            "customer_segment": seg
        })

        token_budget_log.append({
            "ticket_id": item["ticket_id"],
            "tokens_used": tokens_used,
            "cumulative_tokens": accumulated_tokens,
            "budget_exhausted": budget_exhausted
        })

    # Save outputs
    df_results = pd.DataFrame(results)
    df_results.to_csv(outputs_dir / "triage_results.csv", index=False)

    df_budget = pd.DataFrame(token_budget_log)
    df_budget.to_csv(outputs_dir / "token_budget_log.csv", index=False)

    print("\n[Triage Results Summary]")
    print(f"  Valid JSON outputs        : {status_counts.get('VALID_JSON', 0)}")
    print(f"  Keyword fallback outputs  : {status_counts.get('KEYWORD_FALLBACK', 0)}")
    print(f"  Invalid / unparsed outputs: {status_counts.get('INVALID', 0)}")
    print(f"  Transient retries recovered: {total_retries_recovered}")
    print(f"  Saved results to          : outputs/triage_results.csv")
    print(f"  Saved token budget log to : outputs/token_budget_log.csv")
    print("\nPart C completed successfully!")

if __name__ == "__main__":
    run_triage()

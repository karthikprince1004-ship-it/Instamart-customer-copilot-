"""
Part D: Policy RAG Assistant
- 40-word chunker with 10-word overlap (15 reference chunks)
- TF-IDF vectorization and cosine similarity retrieval (top-3 chunks)
- Grounded policy answering using mock_llm (POLICY ASSISTANT)
- Evaluation against data/raw/policy_eval.csv (14 questions: 12 answerable, 2 unanswerable)
- Comparative evaluation with zero overlap (overlap=0) demonstrating chunk boundary failure
"""

import sys
import os
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path resolution: ensure project root and src/ directory are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for directory in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if directory not in sys.path:
        sys.path.insert(0, directory)

import mock_llm

# Ground truth policy documents from assignment
DOCS = {
    "refund_policy.txt": (
        "Refund Policy. A refund can be requested only for orders whose status is completed. "
        "Orders with status cancelled were never charged, so no refund applies to them. "
        "If a completed order is delivered in more than 45 minutes, the customer receives a full refund of the order amount. "
        "If a completed order is delivered in 45 minutes or less, no late-delivery refund is paid. "
        "Refunds up to INR 1000 are approved automatically by the support system. "
        "A refund above INR 1000 must be approved by a human support lead before money is returned. "
        "Approved refunds are credited to the original payment method within 5 working days. "
        "A refund request must be raised within 7 days of the delivery date."
    ),
    "delivery_policy.txt": (
        "Delivery Policy. Standard delivery is promised within 30 minutes of order confirmation in all supported cities. "
        "The supported cities are Chennai, Mumbai, Bengaluru, Delhi, Hyderabad and Pune. "
        "A delivery fee of INR 25 is charged on orders below INR 199, and delivery is free on orders of INR 199 or more. "
        "During heavy rain the promised time may extend to 60 minutes and the app shows a rain notice. "
        "A rider will attempt to contact the customer twice before marking an order as undeliverable. "
        "Electronics items are delivered only after the customer shares a one time password with the rider."
    ),
    "payment_policy.txt": (
        "Payment Policy. Customers can pay using UPI, debit card, credit card or cash on delivery. "
        "Cash on delivery is available only for orders up to INR 2500. "
        "If a payment fails but money is deducted, the amount is returned automatically within 48 hours. "
        "A customer who sees two charges for one order should raise a billing ticket with the order identifier. "
        "Wallet balance can be used for any order and is never refunded as cash. "
        "Coupons cannot be combined with wallet balance on the same order."
    ),
    "privacy_policy.txt": (
        "Privacy Policy. Customer phone numbers and email addresses are used only for order updates and support. "
        "Support tickets are processed with personal details masked before any automated tool reads them. "
        "Delivery addresses are deleted from active systems 180 days after the last order. "
        "A customer can ask for account deletion and the request is completed within 30 days. "
        "Payment card numbers are never stored by the support system."
    ),
    "escalation_policy.txt": (
        "Escalation Policy. A ticket about fraud or an unauthorized transaction is marked high priority and sent to the security team within 2 hours. "
        "Tickets from premium customers are answered within 1 hour, and all other tickets within 24 hours. "
        "If an automated assistant cannot resolve a ticket within 6 steps, it must hand the case to a human agent with a short summary. "
        "A human agent must call the customer back if such a high-value request is rejected."
    )
}

def load_policy_docs() -> dict:
    """Loads policy text files from data/policies/ if present, otherwise uses embedded DOCS."""
    pol_dir = PROJECT_ROOT / "data" / "policies"
    if pol_dir.exists():
        loaded = {}
        for fn in sorted(DOCS.keys()):
            p = pol_dir / fn
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    loaded[fn] = f.read().strip()
        if len(loaded) == 5:
            return loaded
    return DOCS

def chunk_text(text: str, chunk_size: int = 40, overlap: int = 10) -> list[str]:
    """Splits text into chunks of chunk_size words with overlap words."""
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks

class PolicyRAG:
    def __init__(self, chunk_size: int = 40, overlap: int = 10):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.docs = load_policy_docs()
        self.chunks = []

        for fn, text in self.docs.items():
            fn_chunks = chunk_text(text, chunk_size, overlap)
            for c in fn_chunks:
                self.chunks.append({"source": fn, "text": c})

        self.vectorizer = TfidfVectorizer()
        texts = [c["text"] for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Ranks chunks by cosine similarity to query vector."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        ranked_indices = sims.argsort()[::-1][:k]

        results = []
        for idx in ranked_indices:
            results.append({
                "source": self.chunks[idx]["source"],
                "score": float(sims[idx]),
                "text": self.chunks[idx]["text"]
            })
        return results

    def answer(self, question: str, k: int = 3) -> tuple[str, list[str]]:
        """Augments prompt with top-k retrieved chunks and answers via mock_llm."""
        retrieved = self.retrieve(question, k=k)
        sources = [r["source"] for r in retrieved]

        context_parts = []
        for r in retrieved:
            context_parts.append(f"[{r['source']}]\n{r['text']}")
        context_str = "\n\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a POLICY ASSISTANT. Answer only from the supplied context. "
                    "If the answer is not present, say: I could not find that information in the policy documents."
                )
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context_str}\nQUESTION:\n{question}"
            }
        ]

        resp = mock_llm.chat(messages)
        content = resp.choices[0].message.content
        return content, sources

def run_policy_evaluation():
    print("=" * 70)
    print("  INSTAMART COPILOT — PART D: POLICY RAG ASSISTANT EVALUATION")
    print("=" * 70)

    # 1. Chunking verification
    rag_10 = PolicyRAG(chunk_size=40, overlap=10)
    print(f"\n[Task 1: Chunking with 10-word overlap]")
    print(f"  Total chunks generated: {len(rag_10.chunks)} (Expected: 15)")

    counts = {}
    for c in rag_10.chunks:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    for fn, cnt in sorted(counts.items()):
        print(f"    - {fn:22s}: {cnt} chunks")

    # 2. Retrieval verification query
    test_query = "refund policy late delivery"
    top3 = rag_10.retrieve(test_query, k=3)
    top3_sources = [r["source"] for r in top3]
    print(f"\n[Task 2: Retrieval Check for '{test_query}']")
    print(f"  Top 3 sources: {top3_sources}")
    print(f"  All from refund_policy.txt: {all(s == 'refund_policy.txt' for s in top3_sources)}")

    # 3. Load policy_eval.csv questions
    eval_csv_path = PROJECT_ROOT / "data" / "raw" / "policy_eval.csv"
    qa_list = []
    if eval_csv_path.exists():
        df_eval = pd.read_csv(eval_csv_path, keep_default_na=False)
        for _, row in df_eval.iterrows():
            qa_list.append((
                str(row["question"]).strip(),
                str(row["expected_doc"]).strip(),
                str(row["expected_fact"]).strip()
            ))
    else:
        # Fallback to authentic assignment benchmark if CSV not found
        qa_list = [
            ("What delivery time makes a completed order eligible for a full refund?", "refund_policy.txt", "more than 45 minutes"),
            ("Who must approve a refund above INR 1000?", "refund_policy.txt", "human support lead"),
            ("How many days do I have to raise a refund request?", "refund_policy.txt", "within 7 days"),
            ("What is the delivery fee on small orders?", "delivery_policy.txt", "INR 25"),
            ("On orders of what value is delivery free?", "delivery_policy.txt", "INR 199 or more"),
            ("Why does the electronics delivery need a one time password?", "delivery_policy.txt", "one time password"),
            ("Is cash on delivery available for a large order?", "payment_policy.txt", "INR 2500"),
            ("How soon is money returned when a payment fails but is deducted?", "payment_policy.txt", "48 hours"),
            ("Can wallet balance be refunded as cash?", "payment_policy.txt", "never refunded as cash"),
            ("When are delivery addresses deleted from active systems?", "privacy_policy.txt", "180 days"),
            ("How fast must a fraud ticket reach the security team?", "escalation_policy.txt", "within 2 hours"),
            ("After how many steps must an automated assistant hand over to a human?", "escalation_policy.txt", "6 steps"),
            ("What is the favourite colour of the company mascot?", "NONE", ""),
            ("Who won the cricket world cup final?", "NONE", "")
        ]

    # 4. Evaluation helper
    def evaluate(rag_instance, label: str):
        doc_hits = 0
        fact_hits = 0
        fallback_hits = 0
        answerable_count = 0
        unanswerable_count = 0
        broken = []

        for q, exp_doc, exp_fact in qa_list:
            ans, srcs = rag_instance.answer(q)
            if exp_doc and exp_doc != "NONE":
                answerable_count += 1
                if exp_doc in srcs:
                    doc_hits += 1
                if exp_fact in ans:
                    fact_hits += 1
                else:
                    broken.append((q, exp_fact, ans))
            else:
                unanswerable_count += 1
                if ans.startswith("I could not find"):
                    fallback_hits += 1

        print(f"\n[{label}]")
        print(f"  Total Chunks: {len(rag_instance.chunks)}")
        print(f"  Retrieval Recall (Expected Doc in Top-3): {doc_hits}/{answerable_count}")
        print(f"  Grounded Fact Accuracy                : {fact_hits}/{answerable_count}")
        print(f"  Unanswerable Fallback Rate             : {fallback_hits}/{unanswerable_count}")
        if broken:
            print(f"  Broken Question Example on Boundary   :")
            print(f"    Q: {broken[0][0]}")
            print(f"    Expected fact: '{broken[0][1]}'")
            print(f"    Answer given : '{broken[0][2][:80]}...'")

    evaluate(rag_10, "Evaluation with 10-Word Overlap (Production)")

    rag_0 = PolicyRAG(chunk_size=40, overlap=0)
    evaluate(rag_0, "Comparative Evaluation with 0-Word Overlap (Zero Overlap)")

    print("\n" + "=" * 70)
    print("Policy RAG evaluation completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_policy_evaluation()

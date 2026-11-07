"""
Instamart Customer Intelligence & Support Copilot
Master Pipeline Runner (run_project.py)
Executes Parts A through D sequentially and validates required deliverables.
"""

import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

REQUIRED_OUTPUTS = [
    # Part A
    "data/clean/customers.csv",
    "data/clean/products.csv",
    "data/clean/orders.csv",
    "data/instamart.db",
    "sql/queries.sql",
    # Part B
    "outputs/customer_segments.csv",
    "outputs/lapse_model.joblib",
    "outputs/retention_list.csv",
    # Part C
    "outputs/triage_results.csv",
    "outputs/triage_budget_demo.csv",
    # Part D
    "outputs/agent_runs.jsonl",
    "outputs/refund_ledger.jsonl",
]

def run_step(step_name: str, script_path: str):
    print(f"\n{'='*70}")
    print(f"  RUNNING: {step_name} ({script_path})")
    print(f"{'='*70}")
    full_path = ROOT_DIR / script_path
    if not full_path.exists():
        print(f"ERROR: Script not found: {full_path}")
        sys.exit(1)
        
    result = subprocess.run([sys.executable, str(full_path)], cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print(f"\n[FAILED] {step_name} exited with error code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[SUCCESS] {step_name} completed successfully.")

def verify_outputs():
    print(f"\n{'='*70}")
    print("  VERIFYING REQUIRED REPOSITORY OUTPUTS")
    print(f"{'='*70}")
    missing = []
    for item in REQUIRED_OUTPUTS:
        path = ROOT_DIR / item
        if path.exists():
            print(f"  [OK] {item}")
        else:
            print(f"  [MISSING] {item}")
            missing.append(item)
            
    # Check state directory exists
    state_dir = ROOT_DIR / "state"
    if state_dir.exists() and state_dir.is_dir():
        print("  [OK] state/ (directory exists)")
    else:
        print("  [MISSING] state/ directory")
        missing.append("state/")

    print(f"{'-'*70}")
    if missing:
        print(f"Verification FAILED. {len(missing)} item(s) missing:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    else:
        print("ALL REQUIRED DELIVERABLES VERIFIED SUCCESSFULLY! Ready for submission.")

def main():
    print("Starting Instamart Capstone Full Pipeline Execution...")
    
    # 0. Generate Synthetic Data
    run_step("Data Generation", "generate_data.py")
    
    # 1. Part A: Cleaning, SQLite & SQL Queries
    run_step("Part A - Data Pipeline & SQLite", "src/data_pipeline.py")
    
    # 2. Part B: Customer Segmentation & Lapse Prediction
    run_step("Part B - Customer Models & Retention", "src/customer_models.py")
    
    # 3. Part C: Support Ticket Triage with Guardrails
    run_step("Part C - Support Ticket Triage", "src/triage.py")
    
    # 4. Part D: Policy RAG & Refund Agent
    run_step("Part D - Policy RAG", "src/policy_rag.py")
    run_step("Part D - Refund Agent", "src/refund_agent.py")
    
    # 5. Final check
    verify_outputs()

if __name__ == "__main__":
    main()

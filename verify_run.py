from langgraph.checkpoint.sqlite import SqliteSaver
from worker.config import STATE_DB

with SqliteSaver.from_conn_string(str(STATE_DB)) as cp:
    checkpoints = list(cp.list(None))
    latest = checkpoints[0]
    vals = latest.checkpoint['channel_values']
    print("=== VERIFICATION OUTCOME ===")
    print("Outcome:", vals.get('verify_result', {}).get('outcome'))
    print("Message:", vals.get('verify_result', {}).get('message'))
    print("Terminal:", vals.get('terminal'))
    
    print("\n=== PLAN STEPS ===")
    for idx, p in enumerate(vals.get('plan', [])):
        print(f"Step {idx}: done={p.get('done')} | {p.get('description')}")
        
    print("\n=== HISTORY ACTIONS & ARGS ===")
    for h in vals.get('history', []):
        print(f"{h.get('action')}: args={h.get('action_args')} | obs={str(h.get('observation'))[:60]}")

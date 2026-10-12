"""Configuration — loads from .env and provides typed access."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
INVOICE_DIR = Path(os.getenv("INVOICE_DIR", str(BASE_DIR / "invoices")))
SQLITE_DB = Path(os.getenv("SQLITE_DB", str(BASE_DIR / "data" / "erp.db")))
STATE_DB = Path(os.getenv("STATE_DB", str(BASE_DIR / "data" / "agent_state.db")))
AUDIT_LOG = Path(os.getenv("AUDIT_LOG", str(BASE_DIR / "logs" / "audit.jsonl")))

# ── LLM ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Determine which provider to use based on what key is present
if GROQ_API_KEY:
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.8-27b")
    LLM_PROVIDER = "groq"
elif HUGGINGFACE_API_KEY and not OPENAI_API_KEY:
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    LLM_PROVIDER = "huggingface"
else:
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    LLM_PROVIDER = "openai"

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# ── Mock ERP ────────────────────────────────────────────────────────────────
ERP_BASE_URL = os.getenv("ERP_BASE_URL", "http://127.0.0.1:8000")
ERP_USERNAME = os.getenv("ERP_USERNAME", "admin")
ERP_PASSWORD = os.getenv("ERP_PASSWORD", "admin123")

# ── Budget limits ───────────────────────────────────────────────────────────
MAX_STEPS = int(os.getenv("MAX_STEPS", "40"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "100000"))
MAX_COST_USD = float(os.getenv("MAX_COST_USD", "0.50"))
MAX_VERIFY_CYCLES = 3
MAX_REFLECTIONS = 3
MAX_STALE_RETRIES = 2
MAX_READ_RETRIES = 3
MAX_RESUBMIT_RETRIES = 2

# ── Thresholds ──────────────────────────────────────────────────────────────
LOOP_FINGERPRINT_WINDOW = 8
LOOP_REPEAT_LIMIT = 3
PROGRESS_STALL_LIMIT = 5
OCR_CONFIDENCE_THRESHOLD = 0.85
AMOUNT_APPROVAL_THRESHOLD = 100000.0  # above this, require human approval
FUZZY_LABEL_MAX_DISTANCE = 2

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Ensure directories exist ────────────────────────────────────────────────
for d in [INVOICE_DIR, SQLITE_DB.parent, STATE_DB.parent, AUDIT_LOG.parent]:
    d.mkdir(parents=True, exist_ok=True)

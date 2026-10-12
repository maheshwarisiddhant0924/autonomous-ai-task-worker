# Autonomous AI Task Worker

An AI agent that takes a natural language task and autonomously completes it — breaking it down into steps, executing actions via browser/files/APIs, handling errors, and verifying results.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for Playwright browsers)
- An OpenAI-compatible API key (GPT-4o recommended)

### Setup
```bash
# Clone and enter the project
git clone <repo-url>
cd autonomous-ai-worker

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Set environment variables
copy .env.example .env
# Edit .env with your API key
```

### Run
```bash
# Start the mock ERP server (in a separate terminal)
python -m worker.mock_erp.server

# Run a task via CLI
python main.py "Find the latest invoice from TechCorp, extract the amount and due date, enter it into our internal system, and tell me once it is done."

# Run with the live dashboard
streamlit run dashboard.py
```

---

## Architecture

The system follows an **11-stage pipeline** where a task-agnostic orchestration graph delegates domain-specific logic to swappable **Toolkit** implementations.

```
User request → [Router] → [Contract & Plan] → [Discover & Select]
  → [Convert → Extract → Ground → Check → Bind]
  → [ReAct Loop: Act → Policy Gate → ACI Tools → Context Manager]
  → [Verifier] → [Evidence & Termination]
```

### Key Components

| Component | Purpose |
|---|---|
| **Router** | LLM matches natural language to a toolkit + parameters |
| **Contract & Plan** | Creates a validation contract and action checklist |
| **Discovery** | Deterministic file scan, company matching, ordering |
| **Extraction Pipeline** | PDF → structured data with grounding and independent verification |
| **ReAct Agent** | Autonomous thought-action-observation loop |
| **Policy Gate** | Human-in-the-loop for irreversible or ambiguous actions |
| **ACI Tools** | Browser automation with atomic writes and pre-submit guards |
| **Verifier** | Independent database read confirms outcome |
| **Recovery Router** | Deterministic error handlers, LLM reflection as fallback |

### Terminal States
- `SUCCESS_VERIFIED` — outcome confirmed by independent read
- `NEEDS_HUMAN` — paused and resumable; waiting on a person
- `FAILED_WITH_REASON` — gave up after bounded attempts, with evidence

---

## Technical Decisions

### Why LangGraph?
Provides built-in state persistence (`SqliteSaver`), human-in-the-loop interrupts, and conditional graph edges — exactly what an autonomous agent with approval gates needs.

### Why deterministic verification instead of LLM self-check?
The CRITIC paper shows models can't reliably self-verify. Our verifier reads the database directly via SQLite (independent of the browser session) and compares against scoped baselines, not row counts.

### Why tool-less extraction?
The LLM extracts data from document text without any tools. This prevents the model from making side-effecting calls during the extraction phase. Raw document text never enters the ReAct loop — the agent only sees validated, bound values.

### Why atomic submit?
`submit_form()` bundles snapshot → key pre-check → guard → submit as one unit. This prevents duplicate writes (key check), wrong values (pre-submit guard), and ambiguous timeouts (read-only verification before retry).

### Why deterministic recovery before LLM reflection?
Most errors have known cause-effect patterns (stale element → re-snapshot, 409 → key check). Using deterministic handlers first is faster, cheaper, and more reliable. LLM reflection is reserved for truly unexpected situations.

### Why semantic-role grounding?
A grounding check that only verifies "this quote exists in the document" can be fooled by dates/amounts that appear in wrong contexts (issue date returned as due date). Semantic-role checks verify the quote appears near relevant keywords.

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | OpenAI GPT-4o (configurable) |
| Validation | Pydantic v2 |
| Browser automation | Playwright |
| PDF processing | PyMuPDF, Tesseract OCR |
| Mock ERP | FastAPI + SQLite |
| State persistence | SQLite (via SqliteSaver) |
| Dashboard | Streamlit |
| Tracing | Structured JSON audit log |

---

## Known Limitations

- Verification confirms the DB matches a grounded contract, not that the right source document was chosen.
- A document with plausible but wrong values may pass grounding; backstops are reconciliation, amount threshold, semantic-role checks, and human prompt.
- The injection scan is heuristic; it limits blast radius, doesn't eliminate the attack.
- Scanned-document accuracy is bounded by OCR quality, mitigated by second reading and confidence gating.
- Whole-table hashing doesn't scale beyond a prototype.
- Single user and process; concurrency is simulated via test hooks.
- Verification needs backend read access, which real systems often lack.
- Recovery adaptation is within-run only; results vary by run.
- Mock environments only: no MFA, CAPTCHA, or third-party systems.
- Label matching depends on accessible labels in the UI.

---

## What I'd Build Next

- **Label-map inference** via LLM for handling UI changes without config updates
- **Real concurrency model** with proper locking and conflict resolution
- **Second verification channel** (e.g., email confirmation)
- **Multi-document workflows** (e.g., matching POs to invoices)
- **Email-based invoice discovery** (IMAP integration)
- **Larger evaluation sets** with real-world PDF variations

---

## Assumptions

- Invoices are files in a sandbox folder (not email or portal).
- RAG is not used — the task selects one document, not a corpus.
- Backend read access exists for verification (true in mock, not always in production).
- The mock ERP represents a typical CRUD web application.
- AI coding tools (Cursor/Claude) were used during development.

---

## Project Structure

```
autonomous-ai-worker/
├── main.py                     # CLI entry point
├── dashboard.py                # Streamlit live dashboard
├── requirements.txt
├── .env.example
├── worker/
│   ├── __init__.py
│   ├── config.py               # Environment config & constants
│   ├── models.py               # Pydantic models (state, contracts, results)
│   ├── graph.py                # LangGraph orchestration
│   ├── router.py               # Stage 1: Task router
│   ├── planner.py              # Stage 2: Contract & plan
│   ├── discovery.py            # Stage 3: File discovery & selection
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── converter.py        # Stage 4.1: PDF → markdown
│   │   ├── extractor.py        # Stage 4.3: LLM extraction
│   │   ├── grounding.py        # Stage 4.4: Quote & semantic validation
│   │   ├── checker.py          # Stage 4.5: Independent check
│   │   └── binder.py           # Stage 4.7: Bind to contract
│   ├── react/
│   │   ├── __init__.py
│   │   ├── agent.py            # Stage 5: ReAct loop
│   │   ├── policy_gate.py      # Stage 6: Human approval gate
│   │   └── context_manager.py  # Stage 8: Observation compaction
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── browser.py          # Stage 7: Playwright browser tools
│   │   ├── forms.py            # Stage 7: fill_form, submit_form
│   │   └── base.py             # Tool specs & permission tiers
│   ├── verification/
│   │   ├── __init__.py
│   │   └── verifier.py         # Stage 9: Independent DB verification
│   ├── recovery/
│   │   ├── __init__.py
│   │   └── router.py           # Stage 10: Recovery routing
│   ├── evidence.py             # Stage 11: Evidence bundle
│   ├── toolkits/
│   │   ├── __init__.py
│   │   ├── protocol.py         # Toolkit protocol definition
│   │   ├── invoice.py          # Invoice toolkit
│   │   └── trade.py            # Trade toolkit (stretch)
│   └── mock_erp/
│       ├── __init__.py
│       ├── server.py           # FastAPI mock ERP
│       ├── database.py         # SQLite schema & operations
│       └── fault_injection.py  # Test hooks for eval scenarios
├── prompts/
│   ├── router.txt
│   ├── extraction.txt
│   ├── react_system.txt
│   └── reflection.txt
├── invoices/                   # Sandbox invoice folder
│   └── (sample PDFs)
├── tests/
│   ├── __init__.py
│   └── eval_harness.py         # 24-scenario evaluation
└── docs/
    └── pipeline.md             # Full pipeline document
```

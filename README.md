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
git clone https://github.com/maheshwarisiddhant0924/autonomous-ai-task-worker.git
cd autonomous-ai-task-worker

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
# Edit .env with your LLM API key (OpenAI, Groq, or Hugging Face)
```

### Run
```bash
# 1. Start the mock ERP server (in a separate terminal)
python -m worker.mock_erp.server

# 2. Run an end-to-end task via CLI
python main.py "Find the latest invoice from TechCorp, extract the amount and due date, enter it into our internal system, and tell me once it is done."

# 3. Run evaluation harness scenarios (with fault injection & verification)
python evaluate.py 1   # Happy Path (Text PDF -> Extract -> Form Fill -> DB Verify)
python evaluate.py 14  # Fault Recovery: Network timeout on POST
python evaluate.py 22  # Idempotency: Pre-existing identical invoice verified without duplicate write
python evaluate.py 23  # Pre-existing conflict: Blocked before damaging state
python evaluate.py 25  # Generalization: Trade Execution Happy Path
python evaluate.py 26  # Generalization: Trade Duplicate Order Idempotency

# 4. Launch the live observability cockpit
streamlit run dashboard.py
```

---

## Demo Video & Live Walkthrough

- **Video Demonstration**: [Watch Prototype Walkthrough Video](https://github.com/maheshwarisiddhant0924/autonomous-ai-task-worker#demo-video--live-walkthrough) *(Video file / walkthrough recorded in repository documentation)*
- **Interactive Live Demo Instructions**:
  > **Fresh Demo Reset**: If running after previous tests, you can reset all mock ERP data, checkpoints, and screenshots using:
  > - **CLI**: `python reset_demo.py` (or pass `--reset` to `main.py`)
  > - **Dashboard**: Click **"🔄 Reset All State"** in the Streamlit sidebar
  > - **Mock ERP**: Click **"Reset DB"** in the ERP top navigation bar

  1. **Start the local Mock ERP** in Terminal 1:
     ```bash
     python -m worker.mock_erp.server
     ```
  2. **Launch the Streamlit Live Cockpit** in Terminal 2:
     ```bash
     streamlit run dashboard.py
     ```
  3. **Execute an autonomous end-to-end task** in Terminal 3:
     ```bash
     python main.py --reset "Find the latest invoice from TechCorp, extract the amount and due date, enter it into our internal system, and tell me once it is done."
     ```
  4. **Observe Autonomous Execution**: Watch the Playwright browser navigate the ERP, extract and ground values, prompt at policy gates for human confirmation, fill and submit forms atomically, and independently verify the DB state.

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
| **Extraction Pipeline** | PDF/document → structured data with grounding and independent verification |
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

### Why decoupled toolkits (Toolkit Protocol)?
Separating domain-specific extraction, contracts, and verification logic into swappable toolkits allows extending the worker to new tasks (such as financial trade processing via `TradeToolkit`) without rewriting or branching the core 11-stage LangGraph workflow.

### Why deterministic verification instead of LLM self-check?
The CRITIC paper shows models can't reliably self-verify. Our verifier reads the database directly via SQLite (independent of the browser session) and compares against scoped baselines, not row counts.

### Why tool-less extraction?
The LLM extracts data from document text without any tools. This prevents the model from making side-effecting calls during the extraction phase. Raw document text never enters the ReAct loop — the agent only sees validated, bound values.

### Why atomic submit?
`submit_form()` bundles snapshot → key pre-check → guard → submit as one unit. This prevents duplicate writes (key check), wrong values (pre-submit guard), and ambiguous timeouts (read-only verification before retry).

### Why active deterministic recovery with retry budgets?
Rather than passively feeding error messages back into the LLM context, deterministic recovery inspects failure codes (`key_missing`, `wrong_values`, `session_drop`) and directly dispatches appropriate recovery tools (`submit_form`, `fill_form`, `login`), routing immediately via `router_from_recovery`. Bounded per-class retry budgets prevent infinite loops.

### Why exponential backoff with jitter for LLM invocations?
LLM providers (OpenAI, Groq) frequently experience transient rate-limits (TPM/RPM limits) or momentary connection resets. Wrapping all model invocations with `@retry_with_backoff` ensures that transient API drops do not cause high-stakes tasks to fail prematurely.

### Why semantic-role grounding?
A grounding check that only verifies "this quote exists in the document" can be fooled by dates/amounts that appear in wrong contexts (issue date returned as due date). Semantic-role checks verify the quote appears near relevant keywords.

---

## Models, APIs, Frameworks, and External Services Used

### Models
- **OpenAI GPT-4o** (`gpt-4o`): Primary recommended model for multi-step reasoning, structured output generation, and task routing.
- **Groq LLaMA 3.3 70B** (`llama-3.3-70b-versatile`): Fast, low-latency alternative for cloud/local inference.

### APIs & Cloud Services
- **OpenAI API**: Structured extraction, function planning, and ReAct next-step generation.
- **Groq Cloud API**: High-throughput inference backend supported via `config.py`.
- **Hugging Face Inference API**: Optional fallback provider.

### Frameworks & Libraries
- **LangGraph & LangChain Core**: Stateful multi-actor graph orchestration, state checkpointing (`SqliteSaver`), and interrupt-based human-in-the-loop approvals.
- **Playwright**: Headless/headed Chromium browser automation with ARIA accessibility tree snapshots and robust DOM locators.
- **FastAPI & Uvicorn**: Asynchronous local Mock ERP web server providing realistic HTTP endpoints, forms, and session state.
- **Pydantic v2**: Strict typing, data validation, static contract definition, and structured LLM output schemas.
- **PyMuPDF (fitz) & Tesseract OCR (pytesseract)**: Multi-modal document ingestion supporting text-layer extraction and OCR fallbacks for scanned files.
- **Streamlit**: Real-time observability dashboard displaying agent state, audit logs, grounding provenance, and verification outcomes.
- **Levenshtein**: Fuzzy string matching for resilient form label mapping against changing DOMs.
- **Rich**: Terminal output formatting, progress indicators, and interactive CLI prompts.

### Embedded Persistence
- **SQLite3**: Embedded transactional database for both ERP backend records and LangGraph state checkpoints (`.langgraph.sqlite`).

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
autonomous-ai-task-worker/
├── main.py                                      # CLI entry point for natural language tasks
├── evaluate.py                                  # Evaluation harness with 24 scenario mappings & fault injection
├── dashboard.py                                 # Streamlit live observability dashboard & audit cockpit
├── verify_run.py                                # Checkpoint verification inspection utility
├── requirements.txt                             # Python dependencies
├── .env.example                                 # Environment variable template (OpenAI, Groq, HF)
├── Autonomous_AI_Task_Worker_Final_Pipeline.md  # Comprehensive 11-stage architectural specification
├── Autonomous_AI_Task_Worker_Final_Pipeline.pdf # PDF export of the system specification
├── invoices/                                    # Sandbox invoice storage
│   └── TechCorp_Invoice_20261001.pdf           # Sample benchmark invoice
├── worker/
│   ├── __init__.py
│   ├── config.py                                # Environment config, LLM provider routing & constants
│   ├── models.py                                # Pydantic v2 schemas (AgentState, contracts, results)
│   ├── graph.py                                 # LangGraph workflow connecting all 11 stages
│   ├── router.py                                # Stage 1: LLM task router to toolkit
│   ├── discovery.py                             # Stage 3: Deterministic file scan & company matching
│   ├── extraction/                              # Stage 4: Strict data extraction & grounding
│   │   ├── __init__.py
│   │   ├── converter.py                         # PDF text & OCR layer extraction
│   │   ├── extractor.py                         # Tool-less LLM structured extraction
│   │   ├── grounding.py                         # Quote matching & semantic role validation
│   │   ├── checker.py                           # Independent cross-check validation
│   │   └── binder.py                            # Immutable state binding to contract
│   ├── react/                                   # Stage 5 & 6: ReAct execution & policy gating
│   │   ├── __init__.py
│   │   ├── agent.py                             # ReAct loop (Thought -> Action -> Observation)
│   │   ├── policy_gate.py                       # Human approval gate for sensitive/risky actions
│   │   └── context_manager.py                   # History observation compaction
│   ├── tools/                                   # Stage 7: ACI browser & form automation
│   │   ├── __init__.py
│   │   ├── base.py                              # Tool specs, permissions (READ / IDEMPOTENT_WRITE)
│   │   ├── browser.py                           # Playwright session management & ARIA snapshotting
│   │   └── forms.py                             # Form filling, pre-submit guard & atomic submission
│   ├── verification/                            # Stage 9: Independent DB verification
│   │   ├── __init__.py
│   │   └── verifier.py                          # Direct SQLite assertion against grounded contract
│   ├── toolkits/                                # Domain-specific toolkits
│   │   ├── __init__.py
│   │   ├── invoice.py                           # Invoice processing toolkit implementation
│   │   └── trade.py                             # Financial trade order processing toolkit
│   └── mock_erp/                                # Local ERP test environment
│       ├── __init__.py
│       ├── server.py                            # FastAPI mock ERP web application
│       ├── database.py                          # SQLite database schema, operations & unique keys
│       ├── fault_injection.py                   # Simulated network timeouts & test hooks
│       └── templates/                           # HTML templates for mock ERP portal
```

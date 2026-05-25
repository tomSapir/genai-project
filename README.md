<!-- PROJECT LOGO -->
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Logo" width="120" height="120">
</p>

<h1 align="center">SMS Recruitment Chatbot</h1>

<p align="center">
  A multi-agent SMS chatbot that interacts with candidates applying for a Python Developer position — gathers their background, answers questions about the role, and either schedules an interview or politely ends the conversation.
</p>

---
<br></br>

## Table of Contents

- [About The Project](#about-the-project)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Fine-Tuning](#fine-tuning)
- [Code Examples](#code-examples)
- [Project Structure](#project-structure)
- [To-Do List](#to-do-list)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---
<br></br>


## About The Project

> A multi-agent recruitment chatbot built around a **Main Agent** that decides one of three actions every turn — **Continue**, **Schedule**, or **End** — and three specialist **Advisors** that validate or enrich each decision.

<div style="background: #272822; color: #f8f8f2; padding: 10px; border-radius: 8px;">
  <b>Technologies:</b> Python, LangChain, OpenAI (Chat, Embeddings, Fine-Tuning), Chroma, SQLAlchemy, Streamlit
</div>

---
<br></br>


## Features

- [x] Multi-agent orchestration with LangChain
- [x] Three-way action classification — Continue / Schedule / End
- [x] Few-shot prompting on the Main Agent and all Advisors
- [x] Chroma vector search over the Python Developer Job Description
- [x] SQL slot lookup via SQLAlchemy (function-calling pattern)
- [x] Supervised fine-tuning of the Exit Advisor (`gpt-4o-mini`)
- [x] Streamlit chat UI
- [x] End-to-end evaluation with accuracy + confusion matrix
- [x] <span style="color: green; font-weight: bold;">Modular agent package — each advisor lives in its own folder</span>
- [ ] Streamlit Community Cloud deployment _(coming soon!)_

---
<br></br>


## Architecture

Every turn flows through this pipeline:

1. **Main Agent** reads the conversation history and proposes an action (`continue` / `schedule` / `end`) plus a draft SMS reply.
2. The decision is routed to the matching **Advisor** for validation or enrichment:
   - **Exit Advisor** — confirms that ending the conversation is appropriate. Fine-tuned for this binary classification task.
   - **Scheduling Advisor** — queries the SQL slot DB for the three nearest available slots and rewrites the SMS reply with concrete times.
   - **Info Advisor** — retrieves relevant chunks from the Job Description via Chroma and rewrites the SMS reply grounded in those chunks.
3. If an Advisor disagrees with the Main Agent (e.g. the Exit Advisor says *don't end*), the action is demoted back to `continue` and the Info Advisor handles the reply.

---
<br></br>


## Getting Started

### Prerequisites

- Python >= 3.11
- pip
- An OpenAI API key

### Installation

```powershell
git clone https://github.com/tomSapir/genai-project.git
cd genai-project

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Copy the env template and fill in your OpenAI API key:

```powershell
copy .env.example .env
```

Open `.env` and set `OPENAI_API_KEY`. If you have a fine-tuned Exit Advisor model, also set `EXIT_ADVISOR_MODEL=ft:gpt-4o-mini-...` — otherwise leave it commented out and the Exit Advisor will use the base LLM.

### One-time data setup

Build the Chroma vector store from the Job Description PDF (run once):

```powershell
python -c "from app.modules.agents.info_advisor.pdf_embedder import build_vector_store; build_vector_store()"
```

The slot SQLite DB (`data/tech.db`) is created and seeded automatically on first use — no manual step required.

---
<br></br>


## Usage

### Streamlit chat UI

```powershell
streamlit run streamlit_app/streamlit_main.py
```

### Console chat

```powershell
python chat_console.py
```

---
<br></br>


## Evaluation

End-to-end evaluation lives in [`tests/test_evals.ipynb`](tests/test_evals.ipynb). It replays every labeled recruiter turn from `tests/sms_conversations.json` through the deployed multi-agent pipeline and reports **accuracy**, **confusion matrix**, and per-class **precision / recall / F1**.

| metric | value |
|---|---|
| Accuracy | **0.841** (37/44) |
| `schedule` precision / recall | 0.75 / 0.95 |
| `continue` precision / recall | 0.80 / 0.40 |
| `end` precision / recall | 1.00 / 1.00 |

A label-inconsistency analysis cell at the end of the notebook explains why this is at or near the dataset's ceiling.

---
<br></br>


## Fine-Tuning

[`tests/finetune_exit_advisor.ipynb`](tests/finetune_exit_advisor.ipynb) builds train/test JSONL from the labeled dataset, launches a Supervised Fine-Tuning job on `gpt-4o-mini-2024-07-18`, and evaluates the resulting model against the base on a held-out split.

To activate the fine-tuned model in the live pipeline, paste the resulting `ft:...` id into `.env` as:

```env
EXIT_ADVISOR_MODEL=ft:gpt-4o-mini-...
```

`exit_advisor.py` reads this variable and falls back to the base LLM if the fine-tuned model is unreachable.

---
<br></br>


## Code Examples

```python
from app.main import get_bot_response

messages = [
    {"role": "assistant", "content": "Hi, what's your Python background like?"},
    {"role": "user", "content": "I've been writing Python for four years on backend services."},
]

result = get_bot_response(messages)

print(result["action"])    # "schedule"
print(result["response"])  # "Sounds great — could you do Tuesday at 10 AM or Wednesday at 2 PM?"
```

---
<br></br>


## Project Structure

```text
genai-project/
├── app/
│   ├── main.py                            ← get_bot_response entry point
│   └── modules/
│       └── agents/
│           ├── main_agent/main_agent.py
│           ├── exit_advisor/exit_advisor.py
│           ├── scheduling_advisor/
│           │   ├── scheduling_advisor.py
│           │   └── schedule_db.py         ← SQLite mirror of db_Tech.sql
│           └── info_advisor/
│               ├── info_advisor.py
│               └── pdf_embedder.py        ← builds the Chroma store
├── streamlit_app/
│   └── streamlit_main.py
├── data/
│   ├── Python Developer Job Description.pdf
│   ├── db_Tech.sql                        ← teacher-provided SQL spec
│   ├── tech.db                            ← auto-generated SQLite DB
│   └── chroma_db/                         ← built by build_vector_store
├── tests/
│   ├── sms_conversations.json             ← labeled dataset
│   ├── test_evals.ipynb                   ← evaluation notebook
│   └── finetune_exit_advisor.ipynb        ← fine-tuning notebook
├── chat_console.py                        ← CLI chat for local testing
├── requirements.txt
├── requirements.lock
├── .env.example
└── README.md
```

---
<br></br>


## To-Do List

- [x] Multi-agent core (Main + Exit / Scheduling / Info)
- [x] Chroma Job Description retriever
- [x] SQL slot DB with function-calling integration
- [x] Streamlit chat UI
- [x] End-to-end evaluation with accuracy + confusion matrix
- [x] Supervised fine-tuning of the Exit Advisor
- [ ] Streamlit Community Cloud deployment

---
<br></br>


## Contact

**Tom Sapir** — [tom.sapir@akribis-sys.com](mailto:tom.sapir@akribis-sys.com)
Project Link: [https://github.com/tomSapir/genai-project](https://github.com/tomSapir/genai-project)

---
<br></br>


## Acknowledgments

- [LangChain](https://www.langchain.com/)
- [OpenAI](https://platform.openai.com/)
- [Chroma](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [SQLAlchemy](https://www.sqlalchemy.org/)

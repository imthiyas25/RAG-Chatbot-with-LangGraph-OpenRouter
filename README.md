# RAG Chatbot with LangGraph & OpenRouter

This repo holds the Retrieval-Augmented Generation (RAG) prototype we built together. It:

- prepares a complaints dataset (CSV) under `data/`,
- indexes the first 1,000 rows with TF-IDF,
- orchestrates the query → retrieve → generate → format flow via LangGraph, and
- exposes a terminal chatbot that grounds every answer in the retrieved text.

## 1. Prerequisites

- Python 3.12+
- OpenRouter API key (for LLM calls)

## 2. Project Setup

```powershell
# clone repo
git clone https://github.com/<your-user>/rag-chatbot.git
cd rag-chatbot

# create & activate venv (example for PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt  # or run the pip command below
```

Install dependencies inside the virtual environment (either via `requirements.txt` or directly):

```powershell
pip install openai python-dotenv langgraph langchain-core `
    langchain-community langchain-text-splitters pandas `
    scikit-learn kaggle
```

## 3. Dataset

- Place your CSV under `./data/complaints.csv`.  
- Alternatively, drop any `.csv` file into `./data/`; the workflow auto-detects the first CSV and uses the first text column for retrieval.  
- `download_data.py` is included if you want to re-download the Consumer Complaints dataset later (it pulls the files into `data/`).

## 4. Environment Variables

Create a `.env` file (or export variables) with:

```
OPENROUTER_API_KEY=sk-...
```

## 5. Run the Chatbot

```powershell
python rag_workflow.py
```

You’ll see an interactive prompt:

```
You: <ask your question>
```

Type `quit`, `exit`, or `q` to stop. The system always grounds responses in the retrieved dataset text. If the dataset lacks relevant content, the model replies accordingly (no hallucinations).

## 6. Git Workflow Notes

Typical flow:

```powershell
git init
git add .
git commit -m "Initial RAG chatbot setup"
git branch -M main
git remote add origin https://github.com/<your-user>/rag-chatbot.git
git push -u origin main
```

## 7. Files of Interest

- `download_data.py` – downloads and unzips the Kaggle CSV into `./data/`.
- `rag_workflow.py` – complete LangGraph + TF-IDF RAG pipeline with interactive mode.
- `data/` – holds the CSVs (ignored by git by default).

Feel free to swap in new datasets or extend the workflow with additional nodes (e.g., summarization, tool calls, evaluation). Happy hacking!


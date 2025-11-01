# MyFirstRAG
# Automated Issue Remediation 

This repository contains a Streamlit app (`app.py`) that provides a Retrieval-Augmented Generation (RAG) based technical support assistant. It uses OpenAI for embeddings and completions, and MongoDB (with vector search) as the knowledge base.

## What `app.py` does (short)

- Provides a Streamlit UI to: search for solutions, add issues manually to the knowledge base, and bulk-upload issues from a CSV.
- Computes embeddings for issues using OpenAI, stores them in MongoDB documents, and runs vector searches to find similar past issues.
- Generates an adapted solution for a new issue by calling the OpenAI chat/completions API using similar past issues as context.

## Inputs / Outputs (contract)

- Inputs:

  - User-provided issue text (via the UI).
  - CSV file with columns: `issue`, `solution`, `category` (for bulk upload).
  - Environment / sidebar config: OpenAI API key and MongoDB connection string.

- Outputs:

  - A generated solution displayed in the UI.
  - Documents inserted into MongoDB containing: `issue`, `solution`, `category`, `embedding`, and `timestamp`.

- Error modes: missing API key or MongoDB URI, malformed CSV (missing required columns), or connectivity errors to MongoDB/OpenAI.

## Requirements

The repository declares these dependencies in `requirements.txt`:

- streamlit
- pymongo
- openai
- pandas

Install them in your Python environment (recommended in a venv).

## Setup & Run (PowerShell)

1. Create and activate a virtual environment (optional, recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Provide credentials. In the Streamlit sidebar you can paste them interactively, or set them as environment variables:

```powershell
# Set for current session
$env:OPENAI_API_KEY = "sk-..."
$env:MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
```

Note: `app.py` reads the OpenAI key from the environment variable `OPENAI_API_KEY` or from the sidebar input. For MongoDB you must paste the connection string into the sidebar.

4. Run the app:

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).

## Usage notes

- Tabs in the UI:

  - "Find Solution": paste your issue and click "Find Solution" to perform vector search + LLM generation.
  - "Add to Knowledge Base": manually add an issue/solution/category pair.
  - "Upload CSV": upload a CSV with `issue,solution,category` columns and click "Upload to MongoDB".

- The app uses a MongoDB collection (default `issue_remediation.issues`) and expects a vector index named `vector_index` on the `embedding` path. Make sure your MongoDB Atlas / server supports vector search.

- Example CSVs / sample provided by the app: you can download `sample_issues.csv` from the UI. This repo also contains `issue_solution.csv` (if present) that can be uploaded.

## Typical troubleshooting

- "MongoDB connection failed": verify the URI, network access (IP allowlists in Atlas), and username/password.
- "No similar issues found": the KB might be empty or embeddings mismatch; upload a representative CSV and re-run.
- Encoding/Csv errors: save CSV as UTF-8 and ensure the header row includes `issue,solution,category`.
- OpenAI errors: ensure `OPENAI_API_KEY` is valid and has quota.

## Security & privacy

- Do not commit your OpenAI API key or MongoDB credentials to source control.
- Data uploaded to OpenAI (for embeddings/completions) may be logged by the provider — review the OpenAI data usage policy and your organization's policy before sending sensitive data.

## Files of interest

- `app.py` — main Streamlit application (UI, MongoDB integration, embeddings & generation logic).
- `requirements.txt` — Python dependencies.
- `issue_solution.csv` — (if present) example dataset in the workspace.

## Next steps / enhancements

- Add automated tests for CSV parsing and for the `get_embedding` wrapper (mocking OpenAI).
- Add a small script to create the MongoDB vector index automatically.
- Add docs on index creation and Atlas configuration (if using MongoDB Atlas).

---

If you want, I can:

- generate a `README.md` instead of `read.md`, or
- create a short script to bootstrap the MongoDB vector index and show exact commands for Atlas.

Let me know which follow-up you'd like.

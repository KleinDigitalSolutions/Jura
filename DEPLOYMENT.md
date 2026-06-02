# Deployment Runbook

## Local Setup

Use Python 3.12. The project ships `.python-version` for `pyenv`.

```bash
pyenv install 3.12
pyenv local 3.12
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` or `.env.local` with `GEMINI_API_KEY`. The development default is:

```bash
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash-lite
```

`gemini-2.5-flash-lite` is the preferred free-tier development model. DeepSeek and Anthropic can remain as optional paid/fallback providers.

## Rebuild Local Index

Scraping must run locally because some legal sources block datacenter IPs.

```bash
source .venv/bin/activate
python main.py --run-gesetze
python main.py --run-urteile
python main.py --stats
python main.py --search "§ 242 BGB" --gesetz BGB
```

The generated index is stored in `legal_rag_storage/`.

## Modal Setup

Install/authenticate Modal through the Python package:

```bash
source .venv/bin/activate
modal setup
```

Create required secrets:

```bash
modal secret create my-gemini-secret --from-dotenv .env.local --force
modal secret create my-deepseek-secret DEEPSEEK_API_KEY=sk-...
modal secret create my-anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
```

Create/upload the persistent index volume:

```bash
modal volume create legal-rag-data
modal volume put legal-rag-data legal_rag_storage/documents.json documents.json
modal volume put legal-rag-data legal_rag_storage/legal_graph.graphml legal_graph.graphml
modal volume put legal-rag-data legal_rag_storage/qdrant qdrant
```

## Deploy And Smoke Test

```bash
modal deploy modal_deploy.py
curl "https://<your-modal-url>/api/legal/stats"
curl "https://<your-modal-url>/api/legal/search?q=%C2%A7%20242%20BGB&gesetz=BGB"
curl "https://<your-modal-url>/api/legal/ask/enhanced?q=Was%20bedeutet%20Treu%20und%20Glauben%3F&top_k=5"
```

After deployment, verify the chat UI uses streamed enhanced retrieval by checking the `retrieval_method` and citations in the browser network events.

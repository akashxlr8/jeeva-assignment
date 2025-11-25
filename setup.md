# Setup Guide

This document explains how to set up the Persona-Switching Agentic Chatbot project from scratch on a local machine (Windows/Git Bash). It covers environment setup, dependencies, environment variables, database initialization, running the server, and running tests.


## 1. Clone the repository

```bash
https://github.com/akashxlr8/jeeva-assignment.git
cd "jeeva assignment"
```

## 2. Create and activate a virtual environment

Use a virtual environment to isolate dependencies.

For Windows (PowerShell / CMD):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

For Git Bash / WSL (bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
# or
. .venv/Scripts/activate
```

Upgrade pip (recommended):

```bash
pip install --upgrade pip setuptools wheel
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you run into issues with `langgraph` or other packages, ensure your Python version is compatible and consider creating a fresh virtualenv.

## 4. Add environment variables

Create a `.env` file in the project root or export environment variables in your shell. At minimum:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Notes:
- If `OPENAI_API_KEY` is not set, the app falls back to fast simulated responses (useful for offline testing).
- You can create `.env` with the above line. The app uses `python-dotenv` to load environment variables.

## 5. Initialize database(s)

The project uses a few SQLite files:

- `personas.db`: stores persona definitions (seeded automatically)
- `checkpoints.sqlite`: short-term memory / graph checkpoints (created when the graph runs)
- `store.sqlite`: store for longer-term structured memory used by the graph

Seeding personas (optional): the personas DB is initialized automatically when `src.personas` is imported; to explicitly seed manually:

```bash
python -c "from src.personas import init_personas_db; init_personas_db()"
```

After running the server the first time, `checkpoints.sqlite` and `store.sqlite` will be created automatically.

## 6. Run the application

Start the FastAPI server. Two options:

- Using the included `main.py` entrypoint:

```bash
python main.py
```

- Or run `uvicorn` directly (more control):

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The OpenAPI docs will be available at: `http://localhost:8000/docs`

## 7. Quick API sanity checks

- List personas:

```bash
curl "http://localhost:8000/personas"
```

- Send a chat message (example):

```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "user123", "message": "act like my mentor, do you have any advice for optimising for high agency mindset?"}'
```

- Retrieve chat history:

```bash
curl "http://localhost:8000/chat_history?user_id=user123"
```

## 8. Running tests

Unit tests (FastAPI in-process tests):

```bash
pytest tests/test_api.py -q
```

Integration test (spawns the server and runs a sequence of HTTP requests):

```bash
python test_integration.py
```

Notes:
- The integration test depends on `requests` and will start the server. If `OPENAI_API_KEY` is not set the test runs faster because the server uses simulated LLM responses.
- If tests fail with `ModuleNotFoundError: No module named 'langgraph'`, install required packages with `pip install -r requirements.txt` in the activated venv.

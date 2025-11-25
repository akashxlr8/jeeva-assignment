# Persona-Switching Agentic Chatbot

A FastAPI-based chatbot that dynamically switches personas based on user requests, with persistent multi-user and multi-thread conversation storage using SQLite.

## Features
- Dynamic persona switching (e.g., Mentor, Investor)
- Persistent chat history per user and thread
- FastAPI backend with REST endpoints
- LangChain integration with OpenAI GPT-4.1 models

## Setup Instructions

1. Clone or download the repository.

2. Create a virtual environment:
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   - Copy `.env` and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_actual_api_key
     ```

6. Run the application:
   ```
   python main.py
   ```

The API will be available at `http://localhost:8000/docs`.

## API Endpoints

### POST /chat
Send a message and get a response.

**Request Body:**
```json
{
  "user_id": "user123",
  "message": "act like my mentor, how can I scale my product?"
}
```

**Response:**
```json
{
  "response": "As your mentor...",
  "thread_id": "mentor_thread_id",
  "persona": "Mentor"
}
```

### GET /chat_history
Get chat history for a user.

**Query Parameter:** `user_id=user123`

**Response:**
```json
{
  "user_id": "user123",
  "history": {
    "mentor": [
      {"role": "human", "content": "how can I scale?"},
      {"role": "ai", "content": "Focus on..."}
    ],
    "investor": [...]
  }
}
```

### GET /personas
Get the list of available personas.

**Response:**
```json
{
  "personas": ["base", "mentor", "investor", ...]
}
```

## Example cURL Commands

Chat:
```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"user_id": "user123", "message": "act like my mentor"}'
```

History:
```bash
curl "http://localhost:8000/chat_history?user_id=user123"
```

Personas:
```bash
curl "http://localhost:8000/personas"
```

## Future Improvements
- Add more personas (e.g., Coach, Analyst)
- Add authentication and rate limiting
- Deploy to cloud (e.g., AWS, Vercel)
- Implement a frontend UI

## Testing
Run the integration test script to verify the API and persona switching:
```bash
python test_integration.py
```
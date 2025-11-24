[Skip to content](https://www.notion.so/AI-Engineer-Take-Home-Assignment-2a66d0736fb9803b9ee0f64d676a1d7d#main)

# AI Engineer - Take Home Assignment

Design and implement a Persona-Switching Agentic Chatbot that can dynamically adopt different expert personas based on a user's request. The solution must demonstrate a robust architecture for managing state, conversational history, and context switching across multiple, persistent "threads" of conversation.

### Core Requirements

Agentic Framework: Use a recognized agentic framework (e.g., LangGraph, CrewAI, or a custom state machine built with LangChain) to manage the state and logic of the persona-switching mechanism.

Persona Management:

The bot's base system prompt must establish a meta-persona of a "Business Domain Expert" who is capable of assuming more specific roles.

When the user requests a new persona (e.g., "act like my mentor," "be a skeptical investor"), a new, dedicated conversational thread (session) must be initialized.

The agent must adopt the requested persona's specific system prompt within that thread.

Context Switching & Threading:

The user must be able to switch between established threads on the fly (e.g., "now I will do a sales pitch, act like an investor," then later, "back to mentor thread, how about my product scaling?").

The transition must be fast and seamless, with the agent instantly recalling the full context of the previous conversation in that specific thread.

Persistent Long-Term Memory:

Conversations for each thread must be stored persistently (e.g., using a database like SQLite, PostgreSQL, or a key-value store like Redis/S3 for chat history).

This history must be loaded to reconstruct the context whenever a user returns to an existing thread, ensuring long-term memory across user sessions (i.e., if the server restarts or the user logs in later).

LLM Choice: Use any LLM of choice open source or proprietary models.

Deployment & API:

The agent must be exposed via a FastAPI or Flask backend.

Provide a single, simple API endpoint (e.g., /chat) that takes a user\_id and a message

The user\_id 
should be used to manage all threads belonging to a specific user.

Provide a get API endpoint /chat\_history, that takes a user\_id and returns chat history.

### Sample Workflow

|     |     |     |     |
| --- | --- | --- | --- |
| Step | User Query | Expected Bot Action | Key Technical Challenge |
| 1 | act like my mentor | Initialize a new "Mentor" thread. | Thread creation and initial system prompt injection. |
| 2 | how can I scale my product? | Respond as the Mentor, drawing on mentor-specific knowledge. | Standard RAG/LLM call within the Mentor thread. |
| 3 | now I will do a sales pitch, act like an investor | Switch context. Initialize a new "Investor" thread. | Fast context switch, new thread creation. |
| 4 | My product is a B2B SaaS for dog groomers... | Respond as the Investor (e.g., "What's your TAM?"). | LLM call with the Investor persona's prompt. |
| 5 | back to my mentor thread, should I focus on organic growth? | Switch context back to the Mentor thread. Recall the original product scaling discussion from Step 2. | Loading the correct memory/context for the Mentor thread. |

### Time Expectation

Expected time: 2-3 Days

### Submission Requirements

GitHub repository with:

Complete source code

README.md including:

Project setup instructions

Future improvements if any

Requirements.txt or poetry.lock

If deployed or created API endpoints:

API URL

Example cURL commands
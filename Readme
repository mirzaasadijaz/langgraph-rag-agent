# Advanced RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Agents-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)

A robust, stateful AI assistant built with **LangGraph** and **Streamlit** that integrates dynamic Retrieval-Augmented Generation (RAG) with autonomous tool calling. Instead of relying on a single naive generate call, this architecture evaluates queries and dynamically routes them to specialized tools, including local PDF semantic search, live internet browsing, and mathematical evaluation.

Conversational state is maintained across multiple independent sessions using a persistent SQLite checkpointer.

## Key Features

- **Autonomous Tool Calling:** Powered by Groq (`openai/gpt-oss-20b`), the agent decides when to search the web, calculate math, or read uploaded documents based on strict system-prompt guardrails.
- **Persistent Conversational Memory:** All chat threads are saved to a local SQLite database (`chatbot.db`), so users can switch between past conversations from the sidebar.
- **Dynamic Document Ingestion:** Upload PDF files directly through the Streamlit interface. Documents are chunked and embedded with HuggingFace (`all-MiniLM-L6-v2`) into a FAISS vector store for fast semantic retrieval.
- **Live Web Fallback:** Includes both DuckDuckGo and AI-optimized Tavily search tools for real-time data and current events.
- **Mathematical Execution:** Avoids LLM arithmetic hallucinations by routing math expressions through a restricted Python `eval` sandbox.

## System Architecture

| Component | Technology / Library | Description |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Interactive chat interface, session state management, and file uploads. |
| **Orchestration** | LangGraph & LangChain | Manages the conditional agent graph (nodes/edges) and LLM tool binding. |
| **Memory** | SQLite (`SqliteSaver`) | Checkpoints LangGraph state to `chatbot.db` for thread persistence. |
| **Embeddings & Search** | FAISS, PyMuPDF, HuggingFace | Extracts PDF text, splits it into overlapping chunks, and runs similarity search. |
| **Primary LLMs** | Groq & Google GenAI | Fast inference and tool selection using `openai/gpt-oss-20b` and `gemini-3.5-flash-lite`. |

## Prerequisites

- Python 3.8 or higher
- API keys for Groq and Google Gemini
- Tavily API key (optional, but recommended for better search)

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/mirzaasadijaz/advanced-rag-chatbot.git
cd advanced-rag-chatbot
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**

The project requires LangChain ecosystem packages, PyTorch, FAISS, and Streamlit.

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Copy the example file and fill in your keys:

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

Your `.env` should look like this:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

> Never commit your real `.env` file. It is already excluded via `.gitignore`.

## Usage

Launch the app with Streamlit:

```bash
streamlit run streamlit_frontent.py
```

- **Chatting:** Use the main input box to ask questions, solve math problems, or request current news.
- **Document Analysis:** Open the sidebar, upload a `.pdf` file, and wait for the "Indexed!" confirmation. You can then ask questions about the document.
- **Thread Management:** Click **New Chat** in the sidebar to start a fresh conversation, or click a previous chat to resume it.

## Repository Structure

```text
advanced-rag-chatbot/
├── chatbot_backend.py      # LangGraph StateGraph, nodes, conditional edges, SQLite checkpointer
├── streamlit_frontent.py   # Streamlit UI: rendering, sidebar controls, streaming responses
├── tools.py                # @tool definitions: calculator, DuckDuckGo, Tavily, FAISS PDF retriever
├── prompt.py               # SystemMessage with behavior, formatting, and tool-usage rules
├── requirements.txt        # Dependency manifest
├── chatbot.db              # SQLite database for saved chat threads (auto-generated)
├── .env.example            # Template for required environment variables
├── .env                    # Your local API keys (not committed)
├── .gitignore              # Git ignore rules
└── README.md
```

## Author

**Asad Ijaz**

Source code: [mirzaasadijaz/advanced-rag-chatbot](https://github.com/mirzaasadijaz/advanced-rag-chatbot)

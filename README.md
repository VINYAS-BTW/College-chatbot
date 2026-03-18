#  RVITM College Chatbot

A RAG-powered (Retrieval-Augmented Generation) chatbot for RVITM college — answers questions about CSE, AI/ML programs, and admissions using live data scraped from the official website.

---

## Features

-  **RAG Pipeline** — scrapes and indexes RVITM website content into a vector database
-  **Groq LLM** — powered by `llama-3.1-8b-instant` via Groq API for fast responses
-  **Chat History** — remembers previous messages in the conversation
-  **History-Aware Retrieval** — rewrites follow-up questions as standalone queries before retrieval
-  **ChromaDB** — local vector store for fast semantic search
-  **Streamlit UI** — clean, simple chat interface

---

##  Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | FastEmbed |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Web Scraping | LangChain WebBaseLoader + BeautifulSoup |
| Frontend | Streamlit |

---

##  Project Structure

```
chatbot/
├── backend.py       # RAG pipeline — scraping, embeddings, retrieval, LLM chain
├── app.py           # Streamlit frontend
├── .env             # API keys (not pushed to GitHub)
├── .gitignore
├── requirements.txt
└── chroma/          # Vector DB (auto-created, not pushed to GitHub)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/VINYAS-BTW/College-chatbot.git
cd College-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file
```
GROQ_API_KEY=your_groq_api_key_here
USER_AGENT=rvitm-chatbot/1.0
```

Get your free Groq API key at → https://console.groq.com/keys

### 5. Run the chatbot
```bash
streamlit run app.py
```



---

## 💡 How It Works

```
User Question
      ↓
History-Aware Retriever
  → If chat history exists: rewrite question as standalone
  → Retrieve top 10 relevant chunks from ChromaDB (MMR search)
      ↓
Stuff Documents Chain
  → Inject retrieved context into prompt
      ↓
Groq LLM (llama-3.1-8b-instant)
      ↓
Answer
```

---

## 📌 Data Sources

- https://www.rvitm.edu.in/computer-science-engineering/
- https://www.rvitm.edu.in/computer-science-and-engineering-ai-ml/
- https://www.rvitm.edu.in/admission/

---



---

##  Author

**Sai Vinyas**  
GitHub: [@VINYAS-BTW](https://github.com/VINYAS-BTW)

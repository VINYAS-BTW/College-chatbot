import os
from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("USER_AGENT", os.getenv("USER_AGENT", "rvitm-chatbot/1.0"))

from langchain_community.document_loaders import WebBaseLoader
import bs4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda

#scrap
URLS = [
    "https://www.rvitm.edu.in/computer-science-engineering/",
    "https://www.rvitm.edu.in/computer-science-and-engineering-ai-ml/",
    "https://www.rvitm.edu.in/admission/"
]

loader = WebBaseLoader(
    web_paths=URLS,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # Grab ALL content tags, not just headings
            ["h1","h2","h3","h4","h5","h6","p","li","td","th","span","div","table"]
        )
    )
)

docs = loader.load()

#biiger chunk overlaps
splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,       # was 1500 — bigger chunks = more context per retrieval
    chunk_overlap=400      # was 200 — more overlap = less info lost at boundaries
)

chunks = splitter.split_documents(docs)

#embeddings
embeddings = FastEmbedEmbeddings()

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma"
)

# Fetch more chunks — k=10 instead of k=5
retriever = vector_store.as_retriever(
    search_type="mmr",             # MMR = avoids duplicate chunks, more diverse results
    search_kwargs={
        "k": 10,                   # return top 10 chunks
        "fetch_k": 20,             # consider top 20 before MMR filtering
        "lambda_mult": 0.7         # 0=max diversity, 1=max relevance — 0.7 is balanced
    }
)

#llms
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# prompts 

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, "
     "rewrite it as a complete standalone question with full context. "
     "Do NOT answer it, just reformulate if needed."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for RVITM college.\n"
     "Answer the question using ONLY the context provided below.\n"
     "Be detailed and complete — do not cut your answer short.\n"
     "Include all relevant details like fees, seats, eligibility, subjects, etc. if present in context.\n"
     "If the answer is truly not in the context, say 'Sorry! I don't have that information.'\n"
     "Do NOT make up any information.\n\n"
     "Context:\n{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# history retriever
contextualize_chain = contextualize_prompt | llm | StrOutputParser()

history_aware_retriever = RunnableBranch(
    (
        lambda x: not x.get("chat_history"),
        (lambda x: x["input"]) | retriever
    ),
    contextualize_chain | retriever
)

#rag chain
def format_docs(docs):
    # Include source URL in context so LLM knows where info came from
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

rag_chain = (
    RunnablePassthrough.assign(
        context=history_aware_retriever | RunnableLambda(format_docs)
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

#chattt
chat_history = []

def ask_question(query):
    global chat_history

    answer = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

    return answer
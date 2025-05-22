import os
import asyncio
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# Load Gemini API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# Setup ChromaDB 
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("faq_collection")

def add_docs_to_chroma(docs):
    for i, doc in enumerate(docs):
        collection.add(documents=[doc], ids=[f"doc_{i}"])

def query_chroma(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    retrieved_docs = results["documents"][0] if results and results["documents"] else []
    print(f"\n[Debug] Retrieved documents: {retrieved_docs}")
    return retrieved_docs

# Async Gemini generation helper
executor = ThreadPoolExecutor()
async def gemini_async_generate(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, lambda: genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt))
    return response.text.strip()

# RAG Retriever Agent
class RAGRetrieverAgent:
    def __init__(self):
        pass

    async def rag_reply(self, query):
        context_docs = query_chroma(query)
        if not context_docs:
            context_docs = ["No relevant information found in documents."]
        context = "\n".join(context_docs)
        
        prompt = (
            f"You are a helpful assistant.\n"
            f"Use the following context to answer the question clearly and completely.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        print(f"\n Retrieved context:\n{context}\n")
        answer = await gemini_async_generate(prompt)
        return answer

# RoundRobinGroupChat 
class RoundRobinGroupChat:
    def __init__(self, agents):
        self.agents = agents  

    async def run(self, message):
        data = message
        for agent in self.agents:
            data = await agent.rag_reply(data)  
        return data

# Main async loop
async def main():
    print("Chatbot!")

    
    documents = [
        "Gemini is a large language model developed by Google that can understand and generate human-like text.",
        "ChromaDB is a vector database used for storing document embeddings and performing similarity search.",
        "Retrieval-Augmented Generation (RAG) is a technique where a model retrieves relevant documents before answering a question."
    ]
    add_docs_to_chroma(documents)

    rag_agent = RAGRetrieverAgent()
    group_chat = RoundRobinGroupChat([rag_agent])  

    chat_history = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("\nExiting chatbot...")
            break

        answer = await group_chat.run(user_input)
        print(f"\nBot: {answer}\n")

        chat_history.append(f"You: {user_input}\nBot: {answer}\n")

    # Save chat history 
    with open("chat_history.txt", "w", encoding="utf-8") as f:
        f.write("=== Chat History ===\n\n")
        f.writelines(chat_history)
    print("Chat history saved to 'chat_history.txt'.")

if __name__ == "__main__":
    asyncio.run(main())

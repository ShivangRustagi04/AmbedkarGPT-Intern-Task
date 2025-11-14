import os
import sys
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Config
SPEECH_FILE = "speech.txt"
CHROMA_DIR = "chroma_db"
OLLAMA_MODEL = "mistral"  # change this if your Ollama model identifier differs
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 4


def load_text(path: str) -> str:
    if not os.path.exists(path):
        print(f"Error: {path} not found. Put the provided speech text into {path} and retry.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_or_load_vectorstore(text: str, persist_directory: str) -> Chroma:
    """Build a Chroma vectorstore if it doesn't exist, otherwise load it."""
    # Create embeddings object (HF sentence-transformers)
    print("Creating embeddings model (this may download model weights the first time)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # If persistent DB exists, load it
    if os.path.isdir(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing Chroma DB from '{persist_directory}'...")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectordb

    # Otherwise create and persist a new vectorstore
    print("Creating new Chroma DB and indexing chunks...")
    vectordb = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    print(f"Chroma DB created and persisted to '{persist_directory}'.")
    return vectordb


def answer_question(question: str, vectordb: Chroma, llm: Ollama, top_k: int = TOP_K) -> str:
    # Retrieve top-k relevant chunks
    docs = vectordb.similarity_search(question, k=top_k)
    if not docs:
        return "I don't know — no relevant context found."

    context = "\n\n".join([d.page_content for d in docs])

    # Prompt instructing the LLM to *only* use the provided context
    prompt = (
        "You are given context from a short speech. Use ONLY the information in the context to answer the question. "
        "If the answer is not present in the context, reply with: 'I don't know — the answer is not contained in the provided text.'\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer concisely:"
    )

    # Call Ollama LLM
    try:
        resp = llm(prompt)
        # llm(...) usually returns a string for simple wrappers
        return resp
    except Exception as e:
        return f"LLM call failed: {e}"


def main():
    print("Simple RAG CLI over speech.txt — powered by LangChain + Chroma + HF embeddings + Ollama")

    text = load_text(SPEECH_FILE)

    vectordb = build_or_load_vectorstore(text, CHROMA_DIR)

    # Initialize Ollama LLM wrapper
    print(f"Connecting to Ollama model '{OLLAMA_MODEL}'... (ensure Ollama is running locally and model is present)")
    llm = Ollama(model=OLLAMA_MODEL)

    # CLI loop
    print("Ready. Ask questions about the speech (type 'exit' or 'quit' to stop).")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        answer = answer_question(question, vectordb, llm)
        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    main()
# RAG-CLI-Ambedkar — Local Q&A System using LangChain, ChromaDB & Ollama

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A fully local Retrieval-Augmented Generation (RAG) command-line system powered by **LangChain**, **ChromaDB**, **HuggingFace MiniLM embeddings**, and **Ollama Mistral 7B**.  
The system loads Dr. B.R. Ambedkar’s speech, converts it into embeddings, retrieves relevant chunks, and answers user questions **100% offline**.

---

## ⭐ Features

- **Fully Offline** — No API keys, no cloud, 100% local  
- **Embeddings**: MiniLM-L6-v2 (HuggingFace)  
- **Vector Store**: ChromaDB (local persistent database)  
- **LLM**: Mistral 7B running via Ollama  
- **RAG Pipeline**:
  - Load & chunk speech text  
  - Generate embeddings  
  - Store in ChromaDB  
  - Retrieve relevant chunks  
  - Generate answers grounded only in the speech  

---

## 📥 Installation

### 1. Clone the repository
```bash
git clone https://github.com/ShivangRustagi04/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### 2. Create & activate a virtual environment
```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.\.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧩 Install Ollama + Mistral 7B (Required)

### 1. Install Ollama  
Download from:  
https://ollama.com/download

Verify installation:
```bash
ollama --version
```

### 2. Download Mistral 7B model
```bash
ollama pull mistral
```

Test model:
```bash
ollama run mistral
```

---

## ▶️ Running the CLI RAG System

```bash
python main.py
```

Example:
```
Ask a question: What is the real enemy?

Answer:
The belief in the shastras.
```

<img width="576" height="207" alt="image" src="https://github.com/user-attachments/assets/25ee47b0-f8c5-415f-a2c6-57d66aa5f32f" />


To exit:
```
exit
```

---

## 🔧 Project Structure

```
main.py
speech.txt
requirements.txt
chroma_db/   (auto-created)
```

---

## 🧠 How It Works

1. Load text from speech.txt  
2. Split into manageable chunks  
3. Generate embeddings via HuggingFace  
4. Store embeddings in ChromaDB  
5. Retrieve relevant text based on user query  
6. Generate answer using Mistral 7B (via Ollama)  

---

## 📝 Notes
- If you change `speech.txt`, delete `chroma_db/` so embeddings regenerate.  
- Everything runs offline — your data never leaves your system.

---

## 📜 License
MIT License — free for use and modifications.

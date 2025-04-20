# 🧠 Healthcare Awareness Chatbot

An intelligent chatbot that helps users understand and explore various **Indian Government Healthcare Schemes** using Large Language Models (LLMs), **LangChain**, and **Pinecone**.

This project integrates **Retrieval-Augmented Generation (RAG)** with domain-specific healthcare data to provide accurate, real-time answers to natural language questions. It supports two language models:

- ✅ `Flan-T5` (for lightweight and fast deployment)
- ✅ `Mistral 7B` (for richer, more contextual answers in local environments)

---

## 🚀 Live Demo

**💬 Try it on Hugging Face Spaces**  
[👉 Healthcare Chatbot Demo](https://huggingface.co/spaces/your-username/healthcare-chatbot)

---

## 📌 Features

- ❓ Ask any question about Indian Government healthcare schemes.
- 🔍 RAG pipeline: combines LLMs + vector search (via Pinecone).
- 🧠 Supports **Flan-T5** for deployment and **Mistral** for local dev.
- 📊 Built with LangChain for modular, extensible logic.
- 🖼️ Clean UI with background, avatars, and message styling.

---

## 🛠️ Tech Stack

| Tool / Library       | Purpose                                       |
|----------------------|-----------------------------------------------|
| **LangChain**        | RAG pipeline, chain and prompt orchestration  |
| **Hugging Face Transformers** | LLM access (Flan-T5, Mistral)        |
| **Pinecone**         | Vector database for similarity search         |
| **Streamlit**        | Frontend chat interface (for local dev)       |
| **Hugging Face Spaces** | Deployment environment                     |
| **EasyOCR / PIL**    | Image processing support (if needed later)    |

---

## 🧠 Why Two Models?

### 🟢 Flan-T5 (Used for Deployment)

Flan-T5 is a **smaller, fine-tuned T5 model** that provides decent quality for QA tasks. It's ideal for Hugging Face Spaces where:
- RAM is limited (free tier provides ~16GB)
- Inference must be quick
- Hosting large models is not feasible

✅ **Great for cost-effective public demos**

---

### 🟣 Mistral 7B (Used Locally)

Mistral 7B is a **more powerful open-weight LLM**. It offers:
- Better reasoning and fluency
- Handles complex queries better
- Suitable for local or paid GPU environments

💡 This version is provided for those who want **higher performance in a local dev setup**.

---

## 🏗️ Project Structure

```plaintext
Healthcareawareness-Chatbot/
│
├── app_flan_t5.py           # Main app for Hugging Face deployment
├── app_mistral.py           # Local app using Mistral 7B
├── requirements.txt         # Dependencies list
├── README.md                # This file
│
├── data/
│   └── urls.json            # URLs of government healthcare schemes
│
├── src/
│   ├── helper.py            # Functions for loading, parsing text
│   └── storeindex.py        # Pinecone indexing logic
│
├── assets/
│   ├── screenshot_flan.png
│   ├── screenshot_mistral.png
│   └── demo.gif             # Optional: GIF demo of chat
│
└── .gitignore               # Ignore cache, .env, etc.


---


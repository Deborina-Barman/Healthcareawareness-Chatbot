
# 🩺 Healthcare Awareness Chatbot

A Streamlit-based AI chatbot that helps users understand and explore various Indian government healthcare schemes such as **Ayushman Bharat**, **Aarogyasri**, and more. It uses **LangChain**, **Mistral-7B**, and **Pinecone Vector DB** for intelligent Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 🧠 **LLM-Powered Responses**: Answers questions using Mistral-7B-Instruct.
- 📚 **Context-Aware**: Uses RAG with Pinecone and HuggingFace embeddings.
- 🖼️ **Beautiful Streamlit UI**: Clean design with avatars and chat bubbles.
- 🔍 **Citations Included**: Cites sources from uploaded healthcare documents.
- ☁️ **Deployed on Hugging Face Spaces**.

---

## 🧱 Tech Stack

| Tool                  | Usage                            |
|-----------------------|
----------------------------------|
| 🧠 Mistral-7B          | Language model (HuggingFace)     |
| 🦜 LangChain           | Prompt templating, RAG pipeline  |
| 🔍 Pinecone            | Vector search for documents      |
| 🤗 Transformers        | Tokenization & model inference   |
| 🧾 HuggingFace Hub     | Model hosting                    |
| 🧊 Sentence-Transformers | Embeddings                   |
| 🌐 Streamlit           | UI Framework                     |
| 🧪 Unstructured        | Document parsing (PDF)           |

---

## 🛠️ Setup Instructions

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Healthcareawareness-Chatbot.git
cd Healthcareawareness-Chatbot
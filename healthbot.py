import os
import base64
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ⛔️ REMOVE dotenv
# ✅ Use st.secrets for Streamlit Cloud secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]


MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
INDEX_NAME = "healthcare-awareness-index"

st.set_page_config(page_title="Healthcare Awareness Chatbot", layout="centered")

# ✅ Apply background to whole Streamlit app
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

# ✅ Set background
set_background("D:/GENAI/Healthcareawareness-Chatbot/background.jpg")

# ✅ Custom CSS styling
st.markdown("""
    <style>
        .chat-bubble {
            background-color: white;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 1rem;
            border: 1px solid #d9d9d9;
        }
        .user-bubble {
            background-color: #d0f0fd;
            text-align: right;
        }
        .bot-bubble {
            background-color: #ffffff;
            text-align: left;
            border-left: 6px solid #4da6ff;
        }
        .source-link {
            font-size: 0.9rem;
        }
        .avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        .message-row {
            display: flex;
            align-items: flex-start;
            margin-bottom: 1rem;
        }
        h1 {
            text-align: center;
            color: #004d99;
            font-family: 'Arial', sans-serif;
        }
        .caption {
            text-align: center;
            font-size: 1.4rem;
            color: #003366;
            padding: 1rem;
            background-color: #e0f0ff;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        /* Chat input box styling */
        [data-testid="stChatInput"] textarea {
            min-height: 80px !important;
            font-size: 18px !important;
            padding: 12px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Load model
@st.cache_resource
def load_model():
    st.info("🔄 Loading Mistral model, tokenizer, and pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.5)
    return HuggingFacePipeline(pipeline=pipe)

# ✅ Prompt Template
def get_prompt():
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, just say you don't know. Do not make up any answer.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# ✅ Load Pinecone vectorstore
@st.cache_resource
def load_vectorstore():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# ✅ App UI
def main():
    st.markdown("<h1>🩺 Healthcare Awareness Chatbot </h1>", unsafe_allow_html=True)
    st.markdown("<p class='caption'>Ask about government health schemes like <b>Ayushman Bharat</b>, <b>Aarogyasri</b>, etc.</p>", unsafe_allow_html=True)

    # Avatars
    assistant_avatar = r"D:/GENAI/Healthcareawareness-Chatbot/add-user.jpg"
    user_avatar = r"D:/GENAI/Healthcareawareness-Chatbot/user.png"

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Chat history rendering
    for msg in st.session_state.messages:
        avatar = assistant_avatar if msg["role"] == "assistant" else user_avatar
        bubble_class = "bot-bubble" if msg["role"] == "assistant" else "user-bubble"
        username = "Assistant" if msg["role"] == "assistant" else "You"

        with st.container():
            col1, col2 = st.columns([1, 8]) if msg["role"] == "assistant" else st.columns([8, 1])
            if msg["role"] == "assistant":
                col1.image(avatar, width=48)
                col2.markdown(f"<div class='chat-bubble {bubble_class}'><b>{username}</b><br>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                col2.markdown(f"<div class='chat-bubble {bubble_class}'><b>{username}</b><br>{msg['content']}</div>", unsafe_allow_html=True)
                col1.image(avatar, width=48)

    # Chat input
    prompt = st.chat_input("Ask about healthcare schemes...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Generate assistant reply
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        query = st.session_state.messages[-1]["content"]
        with st.spinner("🔄 Getting response..."):
            try:
                llm = load_model()
                vectorstore = load_vectorstore()

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": get_prompt()}
                )

                response = qa_chain.invoke({"query": query})
                result = response["result"]
                sources = response["source_documents"]

                formatted_sources = "<br>".join(
                    f"🔹 <a class='source-link' href='{doc.metadata.get('source', '#')}' target='_blank'>{doc.metadata.get('source', 'Unknown')}</a>"
                    for doc in sources
                )

                full_response = f"{result}<br><br><b>📚 Sources:</b><br>{formatted_sources}"
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()

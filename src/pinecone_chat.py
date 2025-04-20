import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone

# Step 1: Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("🛑 PINECONE_API_KEY is missing in .env")

# Step 2: Constants
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
INDEX_NAME = "healthcare-awareness-index"

# Step 3: Load model
print("🔄 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.5)

# Step 4: Wrap model
llm = HuggingFacePipeline(pipeline=pipe)

# Step 5: Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say you don't know. Do not make up any answer.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def get_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Step 6: Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Step 7: QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": get_prompt()}
)

# Step 8: Chat loop
if __name__ == "__main__":
    while True:
        user_query = input("💬 Ask me about healthcare schemes: ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Have a great day!")
            break
        try:
            response = qa_chain.invoke({"query": user_query})
            print("\n🧠 RESULT:", response["result"])
            print("\n📚 SOURCE DOCUMENTS:")
            for doc in response["source_documents"]:
                print("—", doc.metadata.get("source", "Unknown Source"))
        except Exception as e:
            print("❌ Error:", e)

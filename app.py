import streamlit as st
import os
from dotenv import load_dotenv

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Embeddings & Vector DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -----------------------------
# Load environment / Streamlit secrets
# -----------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("hf", {}).get("token")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("groq", {}).get("api_key")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Groq PDF RAG (Nepali)", layout="wide")
st.title("📄 Groq PDF Q&A (नेपाली)")
st.write("पूर्व-लोड गरिएको PDF embeddings बाट उत्तर दिन्छ।")

# -----------------------------
# Load persisted vectorstore
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # use "cuda" if GPU is available
)

VECTORSTORE_DIR = "./chroma_db"  # must be included in repo

if os.path.exists(VECTORSTORE_DIR):
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    retriever = vectorstore.as_retriever()
    st.success("✅ Vectorstore loaded successfully")
else:
    st.error(f"⚠️ Persisted vectorstore फोल्डर भेटिएन: {VECTORSTORE_DIR}")
    retriever = None

# -----------------------------
# RAG Q&A Loop
# -----------------------------
if retriever:
    if prompt := st.chat_input("पूर्व-लोड गरिएको PDF बारे प्रश्न सोध्नुहोस्..."):
        st.chat_message("user").write(prompt)

        # Initialize Groq LLM
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

        # System prompt enforcing Nepali-only answers
        system_prompt = (
            "तपाईँ एउटा सहायक सहायक हुनुहुन्छ। "
            "सधैं केवल नेपाली भाषामा मात्र जवाफ दिनुहोस्। "
            "कुनै पनि हालतमा हिन्दी वा अन्य भाषा प्रयोग नगर्नुहोस्। "
            "दिइएको प्रसङ्ग (context) प्रयोग गरेर मात्र जवाफ दिनुहोस्। "
            "यदि प्रसङ्ग खाली छ वा सम्बन्धित छैन भने 'मलाई थाहा छैन' भन्नुहोस्। "
            "जवाफ छोटकरीमा दिनुहोस् (अधिकतम ८ वाक्यसम्म)।\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Build chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Prepend explicit Nepali instruction
        user_input = "नेपालीमा जवाफ दिनुहोस्: " + prompt

        # Run RAG
        rag_response = rag_chain.invoke({"input": user_input})
        answer = rag_response["answer"]

        # Show answer
        st.chat_message("assistant").write(answer)

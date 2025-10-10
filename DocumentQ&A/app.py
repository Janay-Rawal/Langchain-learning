import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model = "llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question and context.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

st.title("Document Q&A with Groq LLM and Streamlit")

def vectorEmbedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  #data ingestion
        st.session_state.docs = st.session_state.loader.load() #document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

user_prompt = st.text_input("Enter your question from the documents:")

if st.button("Document Embedding"):
    vectorEmbedding()
    st.write("Document embedding completed.")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    create_retrieval_chain = create_retrieval_chain(retriever,document_chain)
    response = create_retrieval_chain.invoke({"input":user_prompt})
    st.write(response['answer'])

with st.expander("See Source Documents"):
    for i,doc in enumerate(response['context']):
        st.write(doc.page_content)




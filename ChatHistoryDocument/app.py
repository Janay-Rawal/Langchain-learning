import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain.chains import create_retrieval_chain, create_history_aware_retreiver
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory


os.environ["HF_API_TOKEN"] = os.getenv("HF_API_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#streamlit setup
st.title("Conversational RAG with Document and chat history")
st.write("Upload your documents and chat with them!")

api_key = st.write("Enter your Groq API Key:", type="password")
model_select = st.selectbox("Select am model :", ["openai/gpt-oss-120b","openai/gpt-oss-20b","whisper-large-v3-turbo"])
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model = model_select) 

    #Chat interface

    session_id = st.text_input("Session ID:", value="default_session")

    if 'store' not in st.session_state:
        st.session_state['store'] = {}

    uploaded_files = st.file_uploader("Choose a PDF file", type = "pdf",accept_multiple_files=False)
    documents = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFDirectoryLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits,embeddings=embeddings)
        retriever = vectorstore.as_retriever()


    contextualize_q_system_prompt = (
       'Given a chat history and the latest user question which might reference context in the chat history, '
       'formulate a standalone application question which can be understood without the chat history. '
       'Do not answer the question, just reformulate it if needed and otherwise return it as is.'
    )

    contexualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retreiver(llm,retriever,contexualize_q_prompt)









############################################################
## 🧠 Conversational RAG with Chat History – Project Overview
############################################################
##
## This Streamlit app implements a complete Retrieval-Augmented Generation (RAG)
## pipeline that allows users to upload PDFs and chat with their content.
## It combines document retrieval, embeddings, and conversational context memory
## to provide relevant, context-aware responses.
##
############################################################
## ⚙️ Steps Involved
############################################################
##
## 1. Load API Keys and Embeddings
##    - Loads environment variables using `dotenv`.
##    - Sets HuggingFace API token and initializes the embedding model
##      (`sentence-transformers/all-MiniLM-L6-v2`) for converting text chunks into vectors.
##
## 2. Streamlit Interface Setup
##    - Creates a simple UI with fields to:
##        → Enter Groq API key
##        → Select the LLM model
##        → Specify a Session ID
##        → Upload one or more PDF files
##
## 3. Document Upload and Extraction
##    - Saves each uploaded PDF temporarily.
##    - Uses `PyPDFLoader` to extract text from the uploaded files.
##
## 4. Text Chunking
##    - Uses `RecursiveCharacterTextSplitter` to split long documents
##      into overlapping chunks to improve retrieval quality.
##
## 5. Embedding and Vector Store Creation
##    - Converts text chunks into dense vector representations using HuggingFace embeddings.
##    - Stores them in a `Chroma` vector database.
##    - Creates a retriever for semantic search over the embedded text.
##
## 6. History-Aware Question Reformulation
##    - Uses `create_history_aware_retriever()` so that the model can
##      interpret follow-up questions in the context of previous chat turns.
##
## 7. Question Answering with Context
##    - Uses `create_stuff_documents_chain()` to combine retrieved context
##      with the user's question in a prompt.
##    - Builds a final `RAG chain` using `create_retrieval_chain()` that
##      retrieves, grounds, and generates answers.
##
## 8. Persistent Conversation Memory
##    - Uses `RunnableWithMessageHistory` to store user and assistant messages
##      for each session in `st.session_state`.
##    - Enables the chatbot to remember previous turns and continue conversations naturally.
##
## 9. User Interaction
##    - Takes user input from Streamlit.
##    - Invokes the `conversation_rag_chain` to generate context-aware answers.
##    - Displays the assistant’s response and conversation history.
##
############################################################
## ✅ Final Output
############################################################
##
## → A full Conversational RAG Chatbot capable of:
##    - Reading and understanding uploaded PDF files
##    - Retrieving relevant context using vector search
##    - Generating concise, factual answers using the Groq-hosted LLM
##    - Maintaining memory across multiple user turns in a single session
##
## Tech Stack:
##    🧩 LangChain, Streamlit, Chroma, HuggingFace Embeddings, Groq LLM
############################################################

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory


os.environ["HF_API_TOKEN"] = os.getenv("HF_API_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#streamlit setup
st.title("Conversational RAG with Document and chat history")
st.write("Upload your documents and chat with them!")

api_key = st.text_input("Enter your Groq API Key:", type="password")
model_select = st.selectbox("Select am model :", ["openai/gpt-oss-120b","openai/gpt-oss-20b","whisper-large-v3-turbo"])
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model = model_select) 

    #Chat interface

    session_id = st.text_input("Session ID:", value="default_session")

    if 'store' not in st.session_state:
        st.session_state['store'] = {}

    uploaded_files = st.file_uploader("Choose a PDF file", type = "pdf",accept_multiple_files=True)
    documents = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
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
        
        history_aware_retriever = create_history_aware_retriever(llm,retriever,contexualize_q_prompt)
        
        #Answer Question
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        
        def get_session_history(session_id:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your Question:")
        if user_input:
          session_history = get_session_history(session_id)
          response = conversation_rag_chain.invoke(
              {"input": user_input},
              config = {"configurable" : {"session_id": session_id}
                    },
          )
          st.write(st.session_state.store)
          st.write("Assistant:", response['answer'])
          st.write("Chat History: ", session_history.messages)









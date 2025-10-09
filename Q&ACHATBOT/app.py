import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

#Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Groq"


#Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please repond to the user query as best as you can."),
        ("user","Question: {question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):
    llm = ChatGroq(
        groq_api_key=api_key,
        model=llm
    )
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer  = chain.invoke({"question":question})
    return answer

#Streamlit UI
st.title("Q&A Chatbot with Groq")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
llm = st.sidebar.selectbox("Select LLM Model:",["openai/gpt-oss-120b","openai/gpt-oss-20b","whisper-large-v3-turbo"])

#adjust response parameter
temperature = st.sidebar.slider("Temperature:",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens:",min_value=50,max_value=300,value=150)

#main interface for user input
st.write("Ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)

else:
    st.write("Please enter a question to get started.")






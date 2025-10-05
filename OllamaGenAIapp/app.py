import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Langchain tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

#Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user","Question : {question}")
    ]
)

#Streamlit framework

st.title("Langchain demo with Llama3.1")
input_text = st.text_input("What question do you have in mind?")

#ollama llama3.1 model calling
llm = Ollama(model="llama3.1:latest")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser
if input_text:
    response = chain.invoke({"question":input_text})
    st.write(response)
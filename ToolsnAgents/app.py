#https://api.python.langchain.com/en/latest/langchain_api_reference.html#
# 🧠 Streamlit + LangChain + Groq + Agents + Tools: End-to-End App

# 👉 This app uses Streamlit to create a UI chatbot that integrates:
#    - Groq LLMs (like LLaMA 3)
#    - LangChain agents (for reasoning & tool usage)
#    - Tools (Arxiv, Wikipedia, DuckDuckGo) to fetch real-time data

# 🔧 TOOLS: Wrappers like `DuckDuckGoSearchRun`, `WikipediaQueryRun`, `ArxivQueryRun` are created
#           to let the LLM search the web or scientific papers when needed.

# 🧑‍💻 AGENT: We initialize a LangChain `Agent` (specifically CHAT_ZERO_SHOT_REACT_DESCRIPTION),
#            which lets the LLM "think step by step" and decide *which tool to use*.

# 🔑 INPUT: The user types a question in the Streamlit input box (`st.chat_input`)
#    - All messages (user & assistant) are stored in `st.session_state["messages"]`
#    - This maintains the conversation history across interactions.

# 🤖 LLM: We use Groq’s `ChatGroq` LLM with streaming turned on to give real-time feedback.
#        (Model name can be LLaMA-3, Mixtral, Gemma etc.)

# 📟 STREAMLIT CALLBACK: `StreamlitCallbackHandler` shows the thought process of the agent
#     - You can see "Thought", "Action", and "Observation" printed live as the agent works.

# 🔁 FLOW:
#    1. User sends question.
#    2. Agent thinks: Should it answer directly? Use a tool?
#    3. If tool is used → the selected tool fetches data → agent sees result → forms final answer.
#    4. Answer is displayed and stored in the chat history.

# ✅ This is a powerful pattern that enables an LLM to take dynamic action using real-world data.

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler   #helps display the inner workings of an LLM inside a Streamlit app

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.title("🦜🔗 Groq + LangChain + Agents + Tools App")


st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type = "password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content" : "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)   

    llm = ChatGroq(api_key=api_key, model_name = "llama-3.1-8b-instant",streaming=True)
    tools = [search,wiki,arxiv]
    search_agent = initialize_agent(tools,llm,agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True,verbose=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)



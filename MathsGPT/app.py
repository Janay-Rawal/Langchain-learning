import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler

#set up streamlit app
st.set_page_config(page_title="Text to Math problem solver and Data search assistant",page_icon="🦜")
st.title("Text to Math Problem Solver")

groq_api_key = st.sidebar.text_input(label="Groq API key:",type="password")

if not groq_api_key:
    st.info("Please add your GROQ API KEY to continue")
    st.stop()

llm = ChatGroq(model="openai/gpt-oss-20b",groq_api_key=groq_api_key)

#Intialize tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run(),
    description="A tool for searching the internet to find info on topic mentioned."
)

#Initialize Math Tool
math_chain = LLMMathChain.from_llms(llm=llm)
calculator = Tool(
    name = "Calculator",
    func = math_chain.run,
    description = "A tool for answering math related question"
)

prompt = """
You are a agent tasked for solving users mathematical questions.
Logically arrive at the solution and provide a detailed explanation
and display it pointwise for the question below.
Question : {question}
"""

prompt_template = PromptTemplate(
    input_variables = ['question'],
    template = prompt
)

#combine all tools in chain

chain = LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool = Tool(
    name = "Reasoning Tool",
    func = chain.run,
    description = "A tool for answering logic-based and reasoning questions."
)

#Initiailize agents
tools = [wikipedia_tool,calculator,reasoning_tool]
assistant_agent = initialize_agent(tools,llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=False,handle_parsing_errors=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content" : "Hi, I'm a Math chatbot who can answer all maths questions. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
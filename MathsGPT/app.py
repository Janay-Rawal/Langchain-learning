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

llm = ChatGroq(model="qwen/qwen3-32b",groq_api_key=groq_api_key)

#Intialize tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find info on topic mentioned."
)

#Initialize Math Tool
math_chain = LLMMathChain.from_llm(llm=llm)
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

#function to generate response
#def generate_response(question):
#    response = assistant_agent.invoke({'input':question})
#    return response

#Intercation
question = st.text_area("Enter your question:")
if st.button("Find my answer"):
    with st.spinner("Generating response..."):
        if question:
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
            response = assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write("Response:")
            st.success(response)
        else:
            st.warning("Please enter the question")

#I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen of apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruits do I have at the end?

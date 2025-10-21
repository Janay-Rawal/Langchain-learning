import validators, streamlit as st
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain

#streamlit app
st.set_page_config(page_title="Langchain: Summarize text from YT or website", page_icon="🦜" )
st.title("🦜 Langchain: Summarize text from YT or Website")
st.subheader("Summarize URL")

#Get Groq API Key and url to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Enter GROQ API Key:",value="",type="password")

generic_url = st.text_input("URL",label_visibility="collapsed")

llm = ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the follwing content in 300 words
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

if st.button("Summarize the Content from YT or a webiste"):
    #validating
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url")
    else:
        try:
            with st.spinner("Waiting:"):
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"})
                docs = loader.load()

                #chain
                chain = load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")

                

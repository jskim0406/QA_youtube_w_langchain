import streamlit as st 
import databutton as db
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader
from langchain.docstore.document import Document

from langchain.vectorstores import FAISS

from langchain.embeddings import OpenAIEmbeddings


def text_custom(font_size, text):
    '''
    font_size := ['b', 'm', 's']
    '''
    result=f'<p class="{font_size}-font">{text}</p>'
    return result


def main():

    st.set_page_config(
        page_title="Hello, Welcome to Question Answering on Youtube video page",
        layout="wide",  # {wide, centered}
    )
    # reference
    ## https://discuss.streamlit.io/t/change-input-text-font-size/29959/4
    ## https://discuss.streamlit.io/t/change-font-size-in-st-write/7606/2
    st.markdown("""<style>.b-font {font-size:25px !important;}</style>""", unsafe_allow_html=True)    
    st.markdown("""<style>.m-font {font-size:20px !important;}</style>""" , unsafe_allow_html=True)    
    st.markdown("""<style>.s-font {font-size:15px !important;}</style>""" , unsafe_allow_html=True)    
    tabs_font_css = """<style>div[class*="stTextInput"] label {font-size: 15px;color: black;}</style>"""
    st.write(tabs_font_css, unsafe_allow_html=True)

    st.title("Youtube QA Bot")
    t = "Watch all Youtube videos... Sometimes it's hard, right? Just throw us a URL and ask. I'll answer anything. 😋"
    st.markdown(text_custom('m', t), unsafe_allow_html=True)
    st.info('Note: This youtube video itself should have transcript', icon="ℹ️")

    api_key = st.text_input(
        "Enter Open AI Key.",
        placeholder = "sk-...",
        type="password"
    )

    user_in_url = st.text_input(
        "Please enter Youtube URL.",
        value = "https://www.youtube.com/watch?v=o8NPllzkFhE",
    )

    if user_in_url:
        width = 40
        side = max((100 - width) / 2, 0.01)
        _, container, _ = st.columns([side, width, side])
        container.video(data=user_in_url)

    user_question = st.text_input(
            "Please enter your questions in the video.",
            placeholder = "What is the Linux and Why it is created?"
    )

    user_in_lang = st.text_input(
        "Tell us what language the Youtube video is in (For example.. enter 'en' for English or 'ko' for Korean).",
        value = "en",
    )    

    with st.sidebar:
        embeddeing_model = st.selectbox(
            label='Embedding Model',
            options=['text-embedding-ada-002']
        )

        llm_model = st.selectbox(
            label='LLM Model',
            options=["text-davinci-003",
                     "text-curie-001",
                     "text-babbage-001",
                     "text-ada-001"]
        )

        chain = st.radio(
            label='Chain type',
            options=['stuff',
                    'map_reduce',
                    'refine']
        )

        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.7,
        )

        st.markdown(
            """
            **Code:**
            [*Github Repo*](https://github.com/jskim0406/QA_youtube_w_langchain)
            """
        )

    if st.button("Hey ChatGPT. Answer the question right now."):
        API=api_key
        if not API:
            st.warning("Enter your OPENAI API-KEY. If you don't have one Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).")


        # 1. get text data from external source(Youtube video transcription)
        # 참고: https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
        documents = YoutubeLoader.from_youtube_url(user_in_url, language=user_in_lang).load()


        # 2. text preprocessing(Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=['\n\n', '\n', '.', '!', '?', ',', ' ', ''],
            chunk_overlap=200
        )
        docs=text_splitter.split_text(documents[0].page_content)
        new_docs = [Document(page_content=chunk) for chunk in docs]
        

        # 3. define embedding model & provider
        embeddings = OpenAIEmbeddings(openai_api_key=API, model=embeddeing_model)
        
        # 4. create embedding vectorstore(Vector DB) to use as the index
        db = FAISS.from_documents(new_docs, embeddings)

        # 5. Make chin for `question-answering` task with an information retriever
        retriever = db.as_retriever()

        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=API, 
                       model=llm_model, 
                       temperature=temperature, 
                       verbose=True),
            chain_type=chain, 
            retriever=retriever, 
            return_source_documents=True,
            verbose=True)
         
        with st.spinner("Running to answer your question .."):
            query = user_question
            result = qa({"query": user_question})
            st.success(result['result'])


if __name__=='__main__':
    main()
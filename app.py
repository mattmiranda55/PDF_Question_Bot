# watsonx
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
# langchain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# streamlit
import streamlit as st
# other imports
import os
from tempfile import NamedTemporaryFile

st.sidebar.title('Watsonx.AI Keys')
api_key = st.sidebar.text_input("API Key")
projectid = st.sidebar.text_input("Project ID")
pdf_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type=['PDF'])

if not api_key and not projectid and not pdf_files:
    st.write("Please input your API key, Project ID, and PDF files in the sidebar")
else:
    generate_params = {
        GenParams.MAX_NEW_TOKENS: 200
    }

    model = Model(
        model_id = "google/flan-ul2",
        credentials={
            "apikey": api_key,
            "url": "https://us-south.ml.cloud.ibm.com"
        },
        params = generate_params,
        project_id = projectid
    )

    llm = WatsonxLLM(model=model)   

    @st.cache_resource
    def load_pdf(pdf_files):
        loaders = []
        for pdf_file in pdf_files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                loaders.append(UnstructuredPDFLoader(tmp_file.name))
                    
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(),
            text_splitter=CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)).from_loaders(loaders)
        return index

    if pdf_files:
        index = load_pdf(pdf_files)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = 'stuff',
        retriever = index.vectorstore.as_retriever(),
        input_key = 'question'
    )

    st.title('Ask Watsonx questions about your PDFs')

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input('Ask your question here')

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        response = chain.run(prompt)
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content':response})

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import os

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs")
    st.header("Chat with PDFs")
    #st.text_input("Enter your question here:")

    with st.sidebar:
        st.subheader("Your files")
        pdfs =st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Enter"):
            text = extract_pdf_text(pdfs)
            chunk_text = get_text_chunks(text)
            #st.write(chunk_text)
            vectorstore = get_vectorstore(chunk_text)

if __name__ == '__main__':
    main()
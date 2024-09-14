import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
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

def get_conversation_chain(model_name="gemini-pro", temp=0.4):
    prompt_template = """
    Based on the context, provide a detailed answer. Avoid assumptions or incorrect information.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model=model_name, temperature=temp)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    conversation_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return conversation_chain

def process_question(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)
    found_text = db.similarity_search(question)

    chain = get_conversation_chain()

    response = chain({"input_documents": found_text, "question": question}, return_only_outputs=True)

    if "output_text" in response and response["output_text"].strip():
        st.write("Reply: ", response["output_text"])
    else:
        st.write("No relevant answer found for the question.")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs")
    st.header("Chat with PDFs")
    question = st.text_input("Enter your question here:")

    if question:
        process_question(question)

    with st.sidebar:
        st.subheader("Your files")
        pdfs =st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Enter"):
            text = extract_pdf_text(pdfs)
            text_chunks = get_text_chunks(text)
            get_vectorstore(text_chunks)

if __name__ == '__main__':
    main()
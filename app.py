import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Constants
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

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
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain(model_name=CHAT_MODEL, temp=0.4):
    prompt_template = """
    Based on the context, provide a detailed answer. Avoid assumptions or incorrect information.

    Context:{context}
    Question:{question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model=model_name, temperature=temp)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = (
        {"context": lambda x: x["input_documents"], "question": lambda x: x["question"]}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return chain

def process_question(question):
    import os
    
    # Check if FAISS index exists
    if not os.path.exists("faiss_index"):
        st.error("Please upload and process PDF files first before asking questions.")
        return
    
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)
    found_text = db.similarity_search(question)
    
    # Format documents as context string
    context = "\n\n".join([doc.page_content for doc in found_text])

    chain = get_conversation_chain()

    response = chain.invoke({"input_documents": found_text, "question": question})

    if response and response.strip():
        st.write("Reply: ", response)
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
        pdfs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Enter"):
            if not pdfs:
                st.error("Please upload at least one PDF file.")
                return
            
            try:
                st.info("Processing PDFs...")
                text = extract_pdf_text(pdfs)
                
                if not text.strip():
                    st.error("Could not extract text from the PDFs. Please ensure they contain readable text.")
                    return
                
                st.info("Splitting text into chunks...")
                text_chunks = get_text_chunks(text)
                st.info(f"Created {len(text_chunks)} text chunks")
                
                st.info("Creating vector store with embeddings...")
                get_vectorstore(text_chunks)
                st.success("PDFs processed successfully! You can now ask questions.")
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")
                raise

if __name__ == '__main__':
    main()
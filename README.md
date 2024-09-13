# DocuSense

This web app allows users to upload PDF documents and interact with them through a conversational interface. The app uses Google's Generative AI models and FAISS vector storage to extract context from PDFs and answer user questions based on the content.

## Features

- Upload multiple PDF files for analysis.
- Automatically extracts and splits text from PDFs into manageable chunks.
- Uses FAISS (Facebook AI Similarity Search) to store and search the content of the PDFs.
- Utilizes Google Generative AI for embeddings and generating responses.
- A user-friendly Streamlit interface for asking questions and interacting with the PDF content.

## Technologies Used

- **Streamlit**: For the web interface.
- **FAISS**: For vector-based similarity search on the PDF text.
- **Google Generative AI**: To generate embeddings and handle conversational tasks.
- **PyPDF2**: To extract text from PDFs.
- **LangChain**: To chain together language model tasks and processes.
- **dotenv**: To securely load environment variables.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SarangaVP/DocuSense.git

2. **Navigate to the project directory:**
   ```bash
   cd DocuSense

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Set up your environment variables:**
   - Create a .env file in the root directory.
   - Add your Google API key to the .env file:
   ```bash
    GOOGLE_API_KEY=your_google_api_key_here

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```




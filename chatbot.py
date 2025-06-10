import streamlit as st  # For creating the web application interface
from PyPDF2 import PdfReader  # For reading and extracting text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into smaller chunks
import os  # For interacting with the operating system, e.g., accessing environment variables
from dotenv import load_dotenv  # For loading environment variables from a .env file
from langchain_community.vectorstores import Chroma  # Use Chroma instead of FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Add this import
from langchain.chains.question_answering import load_qa_chain  # For loading a question-answering chain
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini chat model import
from langsmith import traceable  # Import traceable for LangSmith tracing

# Load environment variables
load_dotenv()

# Initialize the Gemini chat model with the API key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_output_tokens=1000
)
#
# --- Streamlit Application UI Setup ---

@traceable  # Enable LangSmith tracing for the main app logic
def main():
    st.header("DOC QA Bot")

    with st.sidebar:
        st.title("Upload PDF Files")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # --- Process Uploaded PDF Files ---

    if uploaded_files:
        st.sidebar.success("Files uploaded successfully!")
        for file in uploaded_files:
            st.sidebar.write(file.name)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            length_function=len,
            separators=["\n"]
        )
        documents = []
        for file in uploaded_files:
            pdf_reader = PdfReader(file)
            texts = ""
            for page in pdf_reader.pages:
                texts += page.extract_text() or ""
            if texts:
                chunks = text_splitter.split_text(texts)
                documents.extend(chunks)
        st.sidebar.write(f"Total chunks created: {len(documents)}")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if documents:
            vector_store = Chroma.from_texts(documents, embeddings)
        else:
            vector_store = None
        query = st.text_input("Ask a question about the uploaded documents:")
        if query and vector_store:
            results = vector_store.similarity_search(query)
            chain = load_qa_chain(
                llm=llm,
                chain_type="stuff"
            )
            response = chain.run({"input_documents": results, "question": query})
            st.write("**Response:**", response)
        elif query:
            st.warning("No documents to search. Please upload PDF files first.")
    else:
        st.info("Please upload PDF files to begin.")

if __name__ == "__main__":
    main()
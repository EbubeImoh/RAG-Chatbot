import streamlit as st  # For creating the web application interface
from PyPDF2 import PdfReader  # For reading and extracting text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into smaller chunks
import os  # For interacting with the operating system, e.g., accessing environment variables
from dotenv import load_dotenv  # For loading environment variables from a .env file
from langchain_community.vectorstores import Chroma  # Use Chroma instead of FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Add this import
from langchain.chains.question_answering import load_qa_chain  # For loading a question-answering chain
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini chat model import
import getpass # Ensure the environment variables are loaded from the .env file

# Load environment variables
load_dotenv()

# Set LangSmith and tracing environment variables from .env if present
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
if os.getenv("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")

# Retrieve the Gemini API key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Initialize the Gemini chat model with the API key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_output_tokens=1000
)
#
# --- Streamlit Application UI Setup ---

# Upload PDF files
# Set the main header for the Streamlit application
st.header("DOC QA Bot")

# Define the sidebar content
with st.sidebar:
    st.title("Upload PDF Files")  # Title for the sidebar section
    # Create a file uploader widget in the sidebar for PDF files, allowing multiple uploads
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# --- Process Uploaded PDF Files ---

# Check if any files have been uploaded
if uploaded_files:
    st.sidebar.success("Files uploaded successfully!")  # Display a success message in the sidebar
    # Iterate through the uploaded files and display their names in the sidebar
    for file in uploaded_files:
        st.sidebar.write(file.name)

    # Read and split the PDF files
    # Initialize the text splitter
    # This splitter will divide the text into chunks of 1000 characters
    # with an overlap of 200 characters between chunks.
    # It uses the length of the text as the measure and splits based on newline characters.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        length_function=len,
        separators=["\n"]  # Fixed: should be a list, not a string
    )
    documents = []  # Initialize an empty list to store the text chunks (documents)
    
    # Extract the text
    for file in uploaded_files:
        pdf_reader = PdfReader(file)  # Create a PdfReader object for the current file
        texts = ""  # Initialize an empty string to store the text from the current PDF
        # Iterate through each page of the PDF
        for page in pdf_reader.pages:
            # Extract text from the page and append it to the 'text' string.
            # Use 'or ""' to handle cases where a page might not have extractable text.
            texts += page.extract_text() or ""
        
        # If text was successfully extracted from the PDF
        if texts:
            # Split the extracted text into smaller chunks using the configured text_splitter
            chunks = text_splitter.split_text(texts)
            # Add the generated chunks to the 'documents' list
            documents.extend(chunks)

    # Display the total number of chunks created in the sidebar
    st.sidebar.write(f"Total chunks created: {len(documents)}")
    # st.write(chunks[:5])  # Display first 5 chunks for verification


    # generating embeddings using HuggingFaceEmbeddings wrapper for SentenceTransformer
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # creating vector store
    # Create a Chroma vector store from the processed text documents and their embeddings.
    # This allows for efficient similarity searching among the document chunks.
    if documents:
        vector_store = Chroma.from_texts(documents, embeddings)
    else:
        vector_store = None

    query = st.text_input("Ask a question about the uploaded documents:")

    # --- Do similarity search ---
    if query and vector_store:
        results = vector_store.similarity_search(query)
        # st.write("Top relevant chunks:")
        # for i, res in enumerate(results, 1):
        #     st.write(f"**Chunk {i}:** {res.page_content if hasattr(res, 'page_content') else res}")
        
        # --- output results ---
        chain = load_qa_chain(
            llm=llm,
            # llm=HuggingFaceEmbeddings(model_name="google/gemini-1.5-pro"),
            chain_type="stuff"
        )
        response = chain.run({"input_documents": results, "question": query})
        st.write("**Response:**", response)
    # If there are no documents but a query is provided, show a warning
    elif query:
        st.warning("No documents to search. Please upload PDF files first.")
else:
    st.info("Please upload PDF files to begin.")
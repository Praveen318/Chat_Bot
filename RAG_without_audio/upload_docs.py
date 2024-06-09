from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import shutil

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

def upload_loader_func(pdf_path,persist_directory):
    def loader_func(pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return docs

    # pdf_path = "docs/Oxford Handbook.pdf"  # Specify the path to your PDF file
    docs = loader_func(pdf_path)

    def spliter_func(docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        return text_splitter.split_documents(docs)

    splits = spliter_func(docs)
    print(len(splits))
    persist_directory = persist_directory

    # Delete the existing directory if it exists
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Deleted existing directory: {persist_directory}")

    # Create the embeddings object
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create the new vector database
    vectordb = Chroma.from_documents(
        documents=splits,
        persist_directory=persist_directory,
        embedding=gemini_embeddings
    )

    print(vectordb._collection.count())
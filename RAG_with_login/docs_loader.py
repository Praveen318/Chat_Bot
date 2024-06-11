# # # from langchain_community.document_loaders import PyPDFLoader
# # # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # # from langchain_community.vectorstores import Chroma
# # # from dotenv import load_dotenv
# # # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # # import os


# # # load_dotenv()

# # # os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')





# # # # def loader_func():
# # # #   # from langchain_community.document_loaders import WebBaseLoader
# # # #   # loader = WebBaseLoader('link')
# # # #   # data = loader.load()
# # # #   # Load PDF
# # # #   loaders = [
# # # #       PyPDFLoader("docs\Oxford Handbook.pdf"),
# # # #   ]
# # # #   docs = []
# # # #   for loader in loaders:
# # # #       docs.extend(loader.load())
# # # #   return docs

# # # def loader_func():
# # #     directory = "docs"
# # #     loaders = []

# # #     # List all files in the directory
# # #     for filename in os.listdir(directory):
# # #         # Check if the file is a PDF
# # #         if filename.endswith(".pdf"):
# # #             # Create a PyPDFLoader for each PDF
# # #             loaders.append(PyPDFLoader(os.path.join(directory, filename)))

# # #     docs = []
# # #     for loader in loaders:
# # #         docs.extend(loader.load())

# # #     return docs

# # # docs = loader_func()

# # # def spliter_func(docs):
# # #   # Split
# # #   text_splitter = RecursiveCharacterTextSplitter(
# # #       chunk_size = 1500,
# # #       chunk_overlap = 150
# # #   )
# # #   return text_splitter.split_documents(docs)


# # # splits = spliter_func(docs)
# # # print(len(splits))
# # # persist_directory = "vectordb1"


# # # # If there is no environment variable set for the API key, you can pass the API
# # # # key to the parameter `google_api_key` of the `GoogleGenerativeAIEmbeddings`
# # # # function: `google_api_key = "key"`.

# # # # Access the API key from the environment
# # # # google_api_key = 

# # # # Create the embeddings object
# # # gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # # vectordb = Chroma.from_documents(
# # #     documents=splits,
# # #     persist_directory=persist_directory,
# # #     embedding=gemini_embeddings
# # # )

# # # print(vectordb._collection.count())


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import shutil

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

def loader_func():
    directory = "docs"
    loaders = []

    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a PDF
        if filename.endswith(".pdf"):
            # Create a PyPDFLoader for each PDF
            loaders.append(PyPDFLoader(os.path.join(directory, filename)))

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    return docs

docs = loader_func()

def spliter_func(docs):
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )
    return text_splitter.split_documents(docs)

splits = spliter_func(docs)
print(len(splits))
persist_directory = "vectordb1"

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


# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import os
# import shutil

# load_dotenv()

# os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# def loader_func(pdf_path):
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()
#     return docs

# pdf_path = "docs/Oxford Handbook.pdf"  # Specify the path to your PDF file
# docs = loader_func(pdf_path)

# def spliter_func(docs):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1500,
#         chunk_overlap=150
#     )
#     return text_splitter.split_documents(docs)

# splits = spliter_func(docs)
# print(len(splits))
# persist_directory = "vectordb2"

# # Delete the existing directory if it exists
# if os.path.exists(persist_directory):
#     shutil.rmtree(persist_directory)
#     print(f"Deleted existing directory: {persist_directory}")

# # Create the embeddings object
# gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # Create the new vector database
# vectordb = Chroma.from_documents(
#     documents=splits,
#     persist_directory=persist_directory,
#     embedding=gemini_embeddings
# )

# print(vectordb._collection.count())


# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import os
# import shutil

# load_dotenv()

# os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# def loader_func(web_url):
#     loader = WebBaseLoader(web_url)
#     docs = loader.load()
#     return docs

# web_url = "https://www.jioinstitute.edu.in/faq"  # Specify the URL to your PDF file
# docs = loader_func(web_url)

# def spliter_func(docs):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1500,
#         chunk_overlap=150
#     )
#     return text_splitter.split_documents(docs)

# splits = spliter_func(docs)
# print(len(splits))
# persist_directory = "vectordb3"

# # Delete the existing directory if it exists
# if os.path.exists(persist_directory):
#     shutil.rmtree(persist_directory)
#     print(f"Deleted existing directory: {persist_directory}")

# # Create the embeddings object
# gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # Create the new vector database
# vectordb = Chroma.from_documents(
#     documents=splits,
#     persist_directory=persist_directory,
#     embedding=gemini_embeddings
# )

# print(vectordb._collection.count())

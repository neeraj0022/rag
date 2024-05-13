from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import params
from urllib.parse import quote_plus


# Step 1: Load data from local PDF file
pdf_loader = PyPDFLoader("./files/pdfsam.pdf")
data = pdf_loader.load()

# Step 2: Split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
    "\n\n", "\n", "(?<=\. )", " "], length_function=len)
docs = text_splitter.split_documents(data)
print('Split into ' + str(len(docs)) + ' docs')

# Step 3: Embed data
print("Embeddings Started")
embeddings = OllamaEmbeddings(model="llama2")
print("Embedding Done")


# Step 4: Store data in MongoDB Atlas
# Format the MongoDB connection string with proper escaping
password = "neeraj@22"
escaped_password = quote_plus(password)

uri = f"mongodb+srv://neerajadmin:{escaped_password}@cluster0.lpaxeue.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri)
collection = client[params.db_name][params.collection_name]

# Reset collection without deleting the Search Index
collection.delete_many({})

# Insert documents into MongoDB Atlas with their embeddings
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=params.index_name
)
print("Done..")



import streamlit as st
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import params
from urllib.parse import quote_plus

# Initialize MongoDB python client
from pymongo import MongoClient
password = "neeraj@22"
escaped_password = quote_plus(password)
uri = f"mongodb+srv://neerajadmin:{escaped_password}@cluster0.lpaxeue.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri)
collection = client[params.db_name][params.collection_name]

# Initialize OllamaEmbeddings with LLM2 model
ollama_embeddings = OllamaEmbeddings(model="llama2")

# Initialize vector store
vectorStore = MongoDBAtlasVectorSearch(
    collection, ollama_embeddings, index_name=params.index_name
)

# Initialize Ollama model
ollama = Ollama(model="llama2", temperature=0)
compressor = LLMChainExtractor.from_llm(ollama)

# Initialize Contextual Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorStore.as_retriever()
)
# Streamlit app
st.title("LangChain Q&A with Llama2🦙 + RAG📝")

question = st.text_input("Ask your question:")

import re

def extract_answer_from_ai_response(question, ai_response):
    # Split the AI response into sentences
    sentences = ai_response.split('.')
    
    # Iterate through each sentence to find the one containing the answer
    for sentence in sentences:
        # Check if the question keywords are present in the sentence
        if all(word.lower() in sentence.lower() for word in question.split()):
            # Extract the answer from the sentence (assumes the answer follows the question keywords)
            answer_match = re.search(r'\d+', sentence)  # Match any digits in the sentence
            if answer_match:
                return answer_match.group()  # Return the matched digits as the answer
            else:
                return "Answer not found"
    
    return "Answer not found"


if st.button("Get Answer"):
    if question:
        # Perform similarity search
        docs = vectorStore.max_marginal_relevance_search(question, K=1)
        st.subheader("Query Response:")
        st.write(docs[0].metadata['source'])
        st.write(docs[0].page_content)

        # Perform contextual compression
        compressed_docs = compression_retriever.get_relevant_documents(question)
        st.subheader("AI Response:")
        st.write(compressed_docs[0].metadata['source'])
        
        # Filter the AI response to answer only the specific question
        ai_response = compressed_docs[0].page_content
        answer = extract_answer_from_ai_response(question, ai_response)
        st.write(answer)
    else:
        st.warning("Please enter a question.")

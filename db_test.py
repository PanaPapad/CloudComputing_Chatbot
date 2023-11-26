import getpass
import os
from langchain.document_loaders import TextLoader
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
import pickle
from langchain.chains import VectorDBQA


persist_directory = '/data/CHROMA_DB'
    
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# v = vector_db.similarity_search(q, include_metadata=True)
# chain = load_qa_chain(chat, chain_type="stuff")
# res = chain({"input_documents": v, "question": q})
# print(res["output_text"])

query = "Fogify"
matching_docs = vector_db.similarity_search(query)
print(matching_docs[0])
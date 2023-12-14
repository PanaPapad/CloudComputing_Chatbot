import os
import re
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_transformers import Html2TextTransformer
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader
# urls = "https://ucy-linc-lab.github.io/fogify/"

def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )
    



def remove_extra_spaces(doc):
    doc_page=doc.page_content.strip()
    return " ".join(doc_page.split())
   
os.environ["OPENAI_API_KEY"] = "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"

embeddings = OpenAIEmbeddings()
vectorstore = Chroma("langchain_store", embeddings, persist_directory="./data/CHROMA_DB_4") #3 is stable
vectorstore.persist()

if is_docker():
    docs_dir = "./data/documentation"
else: 
    docs_dir = "./documentation"

# documents=[]
# Read all documents including subdirectories
text_splitter=RecursiveCharacterTextSplitter(chunk_size=3000,
                                               chunk_overlap=1000,
                                               separators=['\n'])
html2text = Html2TextTransformer()

for root, dirs, files in os.walk(docs_dir):
    for file in files:
        # if file.endswith(".xml") or file.endswith(".md"):# or file.endswith(".yaml"):
        #     # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        #     raw_document = TextLoader(os.path.join(root, file)).load()
        #     # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
           

        #     documents = text_splitter.split_documents(raw_document)
        #     vectorstore.add_documents(documents)
            
        if file.endswith('.html') :
            # file_path = os.path.join(root, file)
            # with open(file_path, 'r', encoding='utf-8') as file:
            raw_document = TextLoader(os.path.join(root, file)).load()
            # print(type(raw_document))
            # raw_document=remove_extra_spaces(raw_document)
            documents = text_splitter.split_documents(raw_document)
            documents=html2text.transform_documents(documents)
            # print(documents)
            # for index,doc in enumerate(documents):
            #     doc.page_content=remove_extra_spaces(doc)
                
            vectorstore.add_documents((documents))


        
# make database persistent    
vectorstore.persist()

#test database
print(vectorstore.similarity_search("What is an action and how can i add an action using python code?"))



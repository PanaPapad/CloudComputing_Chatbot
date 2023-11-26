import getpass
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import text
# from langchain.vectorstores import Milvus
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import pickle
os.environ["OPENAI_API_KEY"] = "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"
persist_directory = '/data/CHROMA_DB'


loader = TextLoader("/data/url_content.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
# embeddings_data={'data':embeddings}
# with open('embeddings.pickle', 'wb') as handle:
#     pickle.dump(embeddings_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
vectordb=Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory=persist_directory)
vectordb.persist()



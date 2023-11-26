import getpass
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import text
from langchain.vectorstores import Milvus
from langchain.text_splitter import CharacterTextSplitter

os.environ["OPENAI_API_KEY"] = "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"


loader = TextLoader("url_content.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

docs=vector_db.similarity_search('Fogify')
print(docs[0])
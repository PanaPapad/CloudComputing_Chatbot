from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
import pickle

persist_directory = '/data/CHROMA_DB'
with open('embeddings.pickle', 'rb') as handle:
    embedding = pickle.load(handle)
    
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)
query = "Fogify"
print(qa.run(query))

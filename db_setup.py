import getpass
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import pickle
from langchain.chains import VectorDBQA


# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('data/urlcontent.pickle', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list
    
os.environ["OPENAI_API_KEY"] = "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"
persist_directory = '/data/CHROMA_DB'

documents=read_list()
# loader = TextLoader("/data/url_content.txt")
# documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)
emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    # cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
)

# embeddings = OpenAIEmbeddings()
# embeddings_data={'data':embeddings}
# with open('embeddings.pickle', 'wb') as handle:
#     pickle.dump(embeddings_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
vectordb = Chroma.from_documents(chunks,
                           embedding=embeddings,
                           metadatas=[{"source": f"{i}-wb23"} for i in range(len(chunks))],
                           persist_directory=persist_directory)
# vectordb=Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory=persist_directory)
vectordb.persist()




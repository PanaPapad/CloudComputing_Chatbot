from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain

persist_directory = '/data/CHROMA_DB'
# with open('embeddings.pickle', 'rb') as handle:
#     embedding = pickle.load(handle)
emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    # cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
)

# embeddings = OpenAIEmbeddings()
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)
# query = "Fogify"
# print(qa.run(query))
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
q = "Fogify"
qa = VectorDBQA.from_chain_type(llm=emb_model, chain_type="stuff", vectorstore=vector_db)

print(qa.run(q))
# v = vector_db.similarity_search(q, include_metadata=True)
# chain = load_qa_chain(chat, chain_type="stuff")
# res = chain({"input_documents": v, "question": q})
# print(res["output_text"])
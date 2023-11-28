from getpass import getpass
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
import pickle
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import warnings
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
template = """
Some information about the fogify tool: 
Fogify is an emulation Framework easing the modeling, deployment 
and experimentation of fog testbeds. 
Fogify provides a toolset to: model complex 
fog topologies comprised of heterogeneous resources,
network capabilities and QoS criteria; deploy the modelled
configuration and services using popular containerized 
infrastructure-as-code descriptions to a cloud or local
environment; experiment, measure and evaluate the deployment 
by injecting faults and adapting the configuration at runtime 
to test different "what-if" scenarios that reveal the
limitations of a service before introduced to the public.

Question: {question}

Answer: """

prompt = PromptTemplate(input_variables=['question'], template=template)
repo_id = "gpt2"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 1.0, "max_length": 500}
)

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

def read_list():
    # for reading also binary mode is important
    with open('/data/urlcontent.pickle', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list
    
def read_and_return_DB():
    persist_directory = '/data/CHROMA_DB'
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)  
    return vector_db

def get_relevant_context (user_input,vector_db):None

def setup():
    
    question = "What is fogify?"

    response = llm_chain.run({'question': question})
    return response

def main():
    print("============\nFogify chatbot\n============")
    print("Fogify Chatbot: Hello, how can i help. You can type \"exit\" to exit the chat or ask any question about Fogify")
    
    #read database
    vector_db=read_and_return_DB()
    #warnings.simplefilter("ignore")
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        # context = get_relevant_context(user_input, vector_db)
        # response = get_openai_response(user_input, context)
        # response=create_prompt(user_input)
        response = setup()
        # print response type
        print(response)
        # print("Fogify Chatbot:", response)

if __name__ == "__main__":
    main()

import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def remove_extra_spaces(text):
    return " ".join(text.split())

os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
#load the vector store
vectorstore = Chroma("langchain_store", OpenAIEmbeddings(), persist_directory="./data/CHROMA_DB_2")
#Load GPT-3.5-turbo-1106
llm = OpenAI(model_name='gpt-3.5-turbo-1106')
#Build prompt
template = """
{context}

Question: {question}
"""

#Build LLMChain
prompt = PromptTemplate(input_variables=['question','context'], template=template)
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

def get_relevant_context(question):
    #Get the most similar documents
    docs = vectorstore.similarity_search(question)
    #Build context
    context = ''
    for doc in docs:
        doc.page_content = remove_extra_spaces(doc.page_content)
        if(len(context) + len(doc.page_content) > 6000):
            break
        context += ' '+doc.page_content
    return context




def ask_model(question,context):
    #Ask the model
    response = llm_chain.run({'question': question,'context':context})
    return response


def main():
    print("============\nFogify chatbot\n============")
    print("Fogify Chatbot: Hello, how can i help. You can type \"exit\" to exit the chat or ask any question about Fogify")
    

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        context = get_relevant_context(user_input)
 
        # response = get_openai_response(user_input, context)
        # response=create_prompt(user_input)
        response = ask_model(user_input,context)
        # # print response type
        print(response)
        # print("Fogify Chatbot:", response)

if __name__ == "__main__":
    main()
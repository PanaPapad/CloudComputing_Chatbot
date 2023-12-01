import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# Vectorstore
vectorstore = Chroma("langchain_store", OpenAIEmbeddings(), persist_directory="./data/CHROMA_DB_2")

# Load GPT-3.5-turbo-1106
llm = ChatOpenAI()

# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are chatbot that helps a human with the fogify tool."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        SystemMessagePromptTemplate.from_template(
            "Context: {context}"
        ),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Memory
memory = ConversationBufferMemory(memory_key='chat_history',input_key="question", max_length=3000, return_messages=True)

# LLM Chain
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False, memory=memory)

def remove_extra_spaces(text):
    return " ".join(text.split())

def get_context(question):
    #Get the most similar documents
    docs = vectorstore.similarity_search(question)

    context = ''
    for doc in docs:
        doc.page_content = remove_extra_spaces(doc.page_content)
        if(len(context) + len(doc.page_content) > 6000):
            break
        context += ' '+doc.page_content
    return context

def get_question():
    question = input("Question: ")
    return question

def ask_ai(question):
    context = get_context(question)
    dict = {"question": question, "context": context}
    response = llm_chain(dict)
    print("AI: " + response.get("text"))

def main():
    while True:
        question = get_question()
        if(question == 'exit'):
            break
        ask_ai(question)

if __name__ == "__main__":
    main()


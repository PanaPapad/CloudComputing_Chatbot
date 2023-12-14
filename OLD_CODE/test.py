from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFaceHub
import os




# os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# # def setup():
    
# # synopsis_prompt = PromptTemplate.from_template(
# #     """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
# # Title: {title}
# # Playwright: This is a synopsis for the above play:"""
# # )
# # review_prompt = PromptTemplate.from_template(
# #     """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.
# # Play Synopsis:
# # {synopsis}
# # Review from a New York Times play critic of the above play:"""
# # )
# # os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# # llm = HuggingFaceHub(
# #     repo_id="gpt2", model_kwargs={"temperature": 1.0, "max_length":500}
# # )

# # chain = (
# #     {"synopsis": synopsis_prompt | llm | StrOutputParser()}
# #     | review_prompt
# #     | llm
# #     | StrOutputParser()
# # )
# # chain.invoke({"title": "Tragedy at sunset on the beach"})

# # synopsis_chain = synopsis_prompt | llm | StrOutputParser()
# # review_chain = review_prompt | llm | StrOutputParser()
# # chain = {"synopsis": synopsis_chain} | RunnablePassthrough.assign(review=review_chain)
# # chain.invoke({"title": "Tragedy at sunset on the beach"})
# from langchain.llms import HuggingFaceHub
# llm = HuggingFaceHub(model_kwargs={"temperature": 0.5, "max_length": 500},repo_id="gpt2")
# prompt = "In which country is Tokyo?"
# completion = llm(prompt)
# print(completion)
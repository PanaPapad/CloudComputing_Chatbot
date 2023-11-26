from getpass import getpass
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

HUGGINGFACEHUB_API_TOKEN = getpass()
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
repo_id = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))

# import pickle
# def read_list():
#     # for reading also binary mode is important
#     with open('/data/urlcontent.pickle', 'rb') as fp:
#         n_list = pickle.load(fp)
#         return n_list
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# model_name = 'gpt2'  # Replace with the model you want to use
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# documents=read_list()
# trainingDS = tokenizer.encode(documents, return_tensors='pt')

# from transformers import Trainer, TrainingArguments

# # Define your training arguments
# training_args = TrainingArguments(
#     output_dir='./results',  # Directory to save checkpoints and results
#     num_train_epochs=3,      # Number of epochs
#     per_device_train_batch_size=4,
#     save_steps=500,           # Save model checkpoint every X steps
#     logging_steps=100,        # Log metrics every X steps
#     evaluation_strategy="steps",
#     eval_steps=1000,
#     save_total_limit=2,
# )

# # Define Trainer object
# trainer = Trainer(
#     model=model,                         # The model to be trained
#     args=training_args,                  # Training arguments
#     train_dataset=trainingDS, # Your training dataset
#     tokenizer=tokenizer                   # The tokenizer for encoding
# )

# # Start training
# trainer.train()

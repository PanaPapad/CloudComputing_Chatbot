FROM --platform=linux/amd64 ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update && apt upgrade -y && apt install -y sudo build-essential pip python3 clang curl git
# get Radare
RUN apt update && apt upgrade -y
RUN pip install requests beautifulsoup4 html2text pygments langchain pymilvus openai
RUN pip install sentence_transformers
RUN apt update && apt upgrade -y
# RUN pip install pysqlite3-binary
RUN pip install pickle5
RUN pip install tiktoken
RUN pip install chromadb
ENTRYPOINT /bin/bash

#docker run --privileged -v C:\Users\Panagiotis\Documents\Code\CloudProj\CloudComputing_Chatbot\:/data -it cloudchat /bin/bash

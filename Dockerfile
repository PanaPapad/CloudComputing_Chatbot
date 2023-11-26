FROM --platform=linux/amd64 ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update && apt upgrade -y && apt install -y sudo build-essential clang curl git libboost-all-dev libgmp-dev libpython3-dev libpython3-stdlib llvm-12 llvm-12-dev python3-pip tar && apt-get clean && pip install --upgrade pip && pip3 install Cython lief cmake


# get Radare
RUN apt update && apt upgrade -y

RUN pip install requests beautifulsoup4 html2text pygments langchain pymilvus openai
RUN pip install sentence_transformers
RUN apt update && apt upgrade -y

ENTRYPOINT /bin/bash

#docker run --privileged -v C:\Users\Panagiotis\Documents\Code\CloudProj\CloudComputing_Chatbot\:/data -it cloudchat /bin/bash

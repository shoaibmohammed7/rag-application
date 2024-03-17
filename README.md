# RAG application with GPT 3.5

This is a step-by-step guide to building a simple RAG (Retrieval-Augmented Generation) application using Pinecone and OpenAI's API. The application will allow you to ask questions about any YouTube video.

## Setup

Create a virtual environment and install the required packages:
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

Create an OpenAI API key from [here](https://platform.openai.com/api-keys).<br>
Create a free Pinecone account and get your API key from [here](https://www.pinecone.io/).

Create a `.env` file with the following variables:

```bash
OPENAI_API_KEY = [ENTER YOUR OPENAI API KEY HERE]
PINECONE_API_KEY = [ENTER YOUR PINECONE API KEY HERE]
PINECONE_API_ENV = [ENTER YOUR PINECONE API ENVIRONMENT HERE]
```
Update your required YouTube link in the `rag.py` file.
```bash
YOUTUBE_VIDEO = "ADD_YOUR_YT_VIDEO_LINK_HERE"
```
# 


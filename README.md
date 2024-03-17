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

## High Level Overview of the System Build
![img](https://github.com/shoaibmohammed7/rag-application/assets/55995109/f8683b2a-d96f-478f-8e8e-8ce1e6ef718a)


 __1. Whisper Transcription__ - A component that takes the YouTube video audio and transcribes it into text using OpenAI's Whisper model.<br>
__2. OpenAI Embeddings__ - Converts the chunks of text into vector representations.<br>
__Pinecone Vector Store__ - Stores and retrieves document vectors, creating a searchable index.<br>
__Retriever__ - Retrieves the relevant document vectors based on the user's question.<br>
__Prompt Template__ - A template to format the retrieved context with the question.<br>
__ChatOpenAI Model__ - The model which takes the formatted prompt and generates a response.<br>
__StrOutputParser__ - Parses the model's response into a string.<br>


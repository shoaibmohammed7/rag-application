import os
import tempfile
import whisper
import time
from pytube import YouTube
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_pinecone import PineconeVectorStore


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")


model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
parser = StrOutputParser()

template = """
Answer the question based on the given context. If you don't know the answer, reply "I am not sure".

Context: {context}

Question: {question}
"""


prompt = ChatPromptTemplate.from_template(template)




def update_transcription_if_needed(youtube_video_url):
    metadata_file = "video_metadata.txt"
    transcription_file = "transcription.txt"
    
    # Check if the metadata file exists and read the stored URL
    stored_url = None
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            stored_url = f.read().strip()
    
    if youtube_video_url != stored_url:
        print("Updating transcription for:", youtube_video_url)
        youtube = YouTube(youtube_video_url)
        audio = youtube.streams.filter(only_audio=True).first()
        whisper_model = whisper.load_model("base")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file = audio.download(output_path=tmpdir)

            # Simulate transcription with a loading bar
            print("Transcribing audio, please wait...")
            for _ in tqdm(range(100)):  # Simulating progress
                time.sleep(0.1)  # Sleep to simulate work being done
                # Your transcription logic here
            transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

        with open(transcription_file, 'w') as f:
            f.write(transcription)
        
        with open(metadata_file, 'w') as f:
            f.write(youtube_video_url)
    else:
        print("Using existing transcription for:", youtube_video_url)



# This is the YouTube video we're going to use.
YOUTUBE_VIDEO = "ADD_YOUR_YT_VIDEO_LINK_HERE"


update_transcription_if_needed(YOUTUBE_VIDEO)


# Let's do this only if we haven't created the transcription file yet.
if not os.path.exists("transcription.txt"):
    youtube = YouTube(YOUTUBE_VIDEO)
    audio = youtube.streams.filter(only_audio=True).first()

    # Let's load the model.
    whisper_model = whisper.load_model("base")

    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

        with open("transcription.txt", "w") as file:
            file.write(transcription)



# Load and split the transcribed text from a file.

loader = TextLoader("transcription.txt")
text_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)


# Embeddings are used to convert text into vector representations for similarity comparisons.
model_name = 'text-embedding-ada-002'
embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)


# Create a PineconeVectorStore to store and retrieve document vectors.
index_name = "rag-app"
pinecone = PineconeVectorStore.from_documents(
    documents, embeddings, index_name=index_name
)


# Define the main chain for answering questions.
chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)



while True:
    user_question = input("Please enter your question ('exit' to quit): ").strip()
    if user_question.lower() in ['exit', 'quit']:
        print("Exiting the session.")
        break
    response = chain.invoke(user_question)
    print(f"Response: {response}")
    time.sleep(2)







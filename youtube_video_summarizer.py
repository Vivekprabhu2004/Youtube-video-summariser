

import os
os.environ["GEMINI_API_KEY"] = "AIzaSyDL5hUZH4dlT-imHqjnCj3pKCHUh7P_QTk"

!pip install -q youtube-transcript-api langchain-community langchain-gemini \
               faiss-cpu tiktoken python-dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

video_id = "YvB9jJ42pRY"
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi"])

    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

transcript_list

#text splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

len(chunks)

chunks[5]

!pip install sentence-transformers --quiet

from langchain.embeddings import HuggingFaceEmbeddings

# Indexing (Embedding Generation and Storing in Vector Store)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embeddings)

vector_store.index_to_docstore_id

vector_store.get_by_ids(['1d4405fa-3b55-4594-9daf-e8c42b581a7f'])

#retrival
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever

retriever.invoke('What is indian army')

import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key="AIzaSyDL5hUZH4dlT-imHqjnCj3pKCHUh7P_QTk")

#Agumentation
llm = genai.GenerativeModel(model_name='models/gemini-2.0-flash')

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = "is the topic of indian discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

retrieved_docs

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
context_text

final_prompt = prompt.invoke({"context": context_text, "question": question})

type(final_prompt)

final_prompt_str = final_prompt.to_string()
answer = llm.generate_content(final_prompt_str)
print(answer.text)



import os

# Set your Gemini API key as an environment variable
os.environ["GEMINI_API_KEY"] = "AIzaSyDL5hUZH4dlT-imHqjnCj3pKCHUh7P_QTk"

!pip install -q youtube-transcript-api langchain-community langchain-gemini \
               faiss-cpu tiktoken python-dotenv sentence-transformers

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

video_id = "YvB9jJ42pRY"
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video.")

# text splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

from langchain.embeddings import HuggingFaceEmbeddings

# Indexing (Embedding Generation and Storing in Vector Store)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embeddings)

# retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

import google.generativeai as genai

# Configure Gemini API key (replace with the new key)
genai.configure(api_key="AIzaSyDL5hUZH4dlT-imHqjnCj3pKCHUh7P_QTk")

# Augmentation
llm = genai.GenerativeModel(model_name='models/gemini-2.0-flash')
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.
      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
question = "is the topic of indian discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)
retrieved_docs


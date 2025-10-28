import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id, languages=["hi"]):
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id).find_transcript(languages).fetch()
        transcript = " ".join([chunk.text for chunk in transcript_list])
        return transcript
    except Exception as e:
        print(f"No captions available for this video. Error: {e}")
        return None

def create_vector_store(transcript):
    # Text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Indexing (Embedding Generation and Storing in Vector Store)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def setup_retriever(vector_store, k=4):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever

def generate_answer(question, retriever, llm, prompt_template):
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt_template.invoke({"context": context_text, "question": question})
    final_prompt_str = final_prompt.to_string()
    answer = llm.generate_content(final_prompt_str)
    return answer.text

def generate_summary(transcript):
    llm = genai.GenerativeModel(model_name='models/gemini-2.0-flash')
    prompt = f"Summarize the following transcript in a concise manner:\n\n{transcript}"
    summary = llm.generate_content(prompt).text
    return summary

def answer_question(summary, question):
    llm = genai.GenerativeModel(model_name='models/gemini-2.0-flash')
    prompt = f"Based on the following summary, answer the question:\n\nSummary: {summary}\n\nQuestion: {question}"
    answer = llm.generate_content(prompt).text
    return answer

def process_video(url):
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL. Please provide a valid YouTube video URL."

    transcript = get_transcript(video_id)
    if not transcript:
        return "No captions available for this video. Unable to generate summary."

    summary = generate_summary(transcript)
    return summary

if __name__ == "__main__":
    video_id = "YvB9jJ42pRY"
    transcript = get_transcript(video_id)
    if transcript:
        print("Transcript fetched successfully.")
        vector_store = create_vector_store(transcript)
        retriever = setup_retriever(vector_store)

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
            input_variables=['context', 'question']
        )

        question = "is the topic of indian discussed in this video? if yes then what was discussed"
        answer = generate_answer(question, retriever, llm, prompt)
        print("Answer:", answer)
    else:
        print("Failed to fetch transcript.")

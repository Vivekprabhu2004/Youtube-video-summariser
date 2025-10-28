# YouTube Video Summarizer & Q&A

A web application that generates summaries of YouTube videos and allows users to ask questions based on those summaries.

## Features

- **Video Summary Generation**: Extract transcripts from YouTube videos and generate concise summaries
- **Interactive Q&A**: Ask questions about the video content based on the generated summary
- **Web Interface**: User-friendly Gradio interface with two tabs for summary generation and Q&A
- **Multi-language Support**: Supports Hindi and other languages for transcript extraction

## Technologies Used

- **Backend**: Python, LangChain, Google Gemini AI
- **Frontend**: Gradio
- **Transcript API**: YouTube Transcript API
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Store**: FAISS

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/Vivekprabhu2004/Youtube-video-summariser.git
cd Youtube-video-summariser
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Google Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
python app.py
```

The app will be available at `http://127.0.0.1:7860`

## Deployment Options

### 1. Hugging Face Spaces (Recommended)

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Connect your GitHub repository
3. Set the following in your Space settings:
   - **SDK**: Gradio
   - **App file**: `app.py`
   - **Requirements file**: `requirements.txt`
4. Add your `GEMINI_API_KEY` as a secret in the Space settings
5. Deploy!

### 2. Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub repo
4. Set the main file path to `app.py`
5. Add environment variables in the app settings
6. Deploy

### 3. Railway

1. Connect your GitHub repo to [Railway](https://railway.app/)
2. Add environment variables
3. Deploy

### 4. Render

1. Connect your GitHub repo to [Render](https://render.com/)
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`
4. Add environment variables
5. Deploy

## Usage

1. **Generate Summary**: Paste a YouTube URL and click "Generate Summary" to get a concise summary of the video
2. **Ask Questions**: After generating a summary, switch to the Q&A tab and ask questions about the video content

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)

## Contributing

Feel free to open issues and pull requests!

## License

MIT License

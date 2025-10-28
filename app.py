import gradio as gr
from main import process_video, answer_question

# Global variable to store the current summary
current_summary = None

def generate_summary_interface(youtube_url):
    global current_summary
    if not youtube_url.strip():
        return "Please enter a YouTube URL."

    try:
        summary = process_video(youtube_url)
        current_summary = summary
        return summary
    except Exception as e:
        return f"An error occurred: {str(e)}"

def ask_question_interface(question):
    global current_summary
    if current_summary is None:
        return "Please generate a summary first by entering a YouTube URL."
    if not question.strip():
        return "Please enter a question."

    try:
        answer = answer_question(current_summary, question)
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create Gradio interface with tabs
with gr.Blocks(title="YouTube Video Summarizer & Q&A") as iface:
    gr.Markdown("# YouTube Video Summarizer & Q&A")
    gr.Markdown("First, generate a summary of the video, then ask questions based on that summary.")

    with gr.Tab("Generate Summary"):
        youtube_url = gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video URL here...")
        summary_output = gr.Textbox(label="Summary", lines=10, interactive=False)
        generate_btn = gr.Button("Generate Summary")
        generate_btn.click(generate_summary_interface, inputs=youtube_url, outputs=summary_output)

    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Question", placeholder="Enter your question about the summary...")
        answer_output = gr.Textbox(label="Answer", lines=5, interactive=False)
        ask_btn = gr.Button("Ask Question")
        ask_btn.click(ask_question_interface, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    iface.launch()

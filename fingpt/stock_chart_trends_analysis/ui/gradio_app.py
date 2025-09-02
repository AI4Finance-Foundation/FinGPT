import gradio as gr
from chatbot.chat_handler import handle_user_turn
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DOC_EXT = {".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".htm", ".pptx"}

# --- UI ---
def create_app():
    with gr.Blocks(title="Unified Finance Chatbot (OpenAI)") as demo:
        chat = gr.Chatbot(label=None, height=520, show_copy_button=True, bubble_full_width=False)
        # Note: depending on your Gradio version, MultimodalTextbox signature may vary.
        mm = gr.MultimodalTextbox(
            placeholder="Ask a question, drop a chart image, or upload a document...",
            show_label=False,
            file_types=list(IMG_EXT | DOC_EXT),
            autofocus=True,
            submit_btn=True
        )
        summary_out = gr.Markdown(visible=False)
        sentiment_out = gr.Textbox(visible=False)
        forecast_out = gr.Markdown(visible=False)
        json_out = gr.Code(language="json", visible=False)
        news_out = gr.Dataframe(visible=False)
        json_download = gr.DownloadButton(label="Download final_output.json", visible=False)
        rag_state = gr.State([])
        chat_state = gr.State([])

        def on_mm_submit_pairs(history_pairs, data, rag_sessions, do_news=True):
            user_text = (data or {}).get("text") or ""
            user_files = (data or {}).get("files") or []
            result = handle_user_turn(history_pairs, user_text, user_files, rag_sessions, do_news)
            return (*result, gr.update(value=None))

        mm.submit(
            fn=on_mm_submit_pairs,
            inputs=[chat_state, mm, rag_state],
            outputs=[chat, summary_out, sentiment_out, forecast_out, json_out, news_out, json_download, mm]
        ).then(
            fn=lambda jo, df, jp: (gr.update(value=jo), df, jp),
            inputs=[json_out, news_out, json_download],
            outputs=[json_out, news_out, json_download]
        ).then(
            fn=lambda h, rs: (h, rs),
            inputs=[chat, rag_state],
            outputs=[chat_state, rag_state]
        )

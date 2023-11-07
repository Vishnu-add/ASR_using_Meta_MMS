import gradio as gr
from asr import transcribe,detect_language,transcribe_lang

demo = gr.Interface(transcribe,
                   gr.Audio(source="microphone", label="Use mic"),
                   outputs=["text","text"])
demo2 = gr.Interface(detect_language,
                   gr.Audio(source="microphone", label="Use mic"),
                   outputs=["text","text"])
demo3 = gr.Interface(transcribe_lang,
                   inputs=[gr.Audio(source="microphone", label="Use mic"),"text"],
                   outputs=["text","text"])

tabbed_interface = gr.TabbedInterface([demo,demo2,demo3],["Transcribe by auto detecting language","Detect language","Transcribe by providing language"])

with gr.Blocks() as asr:
    tabbed_interface.render()

if __name__ == "__main__":
    asr.queue(concurrency_count=3)
    asr.launch()

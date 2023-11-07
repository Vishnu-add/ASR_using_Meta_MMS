import gradio as gr
from asr import transcribe,detect_language,transcribe_lang

demo = gr.Interface(transcribe,
                   inputs = "microphone",
                   # gr.Audio(sources=["microphone"]),
                   outputs=["text","text"],
                   examples=["./Samples/Hindi_1.mp3","./Samples/Hindi_2.mp3","./Samples/Tamil_1.mp3","./Samples/Tamil_2.mp3","./Samples/Marathi_1.mp3","./Samples/Marathi_2.mp3","./Samples/Nepal_1.mp3","./Samples/Nepal_2.mp3","./Samples/Telugu_1.wav","./Samples/Telugu_2.wav","./Samples/Malayalam_1.wav","./Samples/Malayalam_2.wav","./Samples/Gujarati_1.wav","./Samples/Gujarati_2.wav","./Samples/Bengali_1.wav","./Samples/Bengali_2.wav"]
)
demo2 = gr.Interface(detect_language,
                     inputs = "microphone",
                   # gr.Audio(sources=["microphone"]),
                   outputs=["text","text"],
                     examples=["./Samples/Hindi_1.mp3","./Samples/Hindi_2.mp3","./Samples/Tamil_1.mp3","./Samples/Tamil_2.mp3","./Samples/Marathi_1.mp3","./Samples/Marathi_2.mp3","./Samples/Nepal_1.mp3","./Samples/Nepal_2.mp3","./Samples/Telugu_1.wav","./Samples/Telugu_2.wav","./Samples/Malayalam_1.wav","./Samples/Malayalam_2.wav","./Samples/Gujarati_1.wav","./Samples/Gujarati_2.wav","./Samples/Bengali_1.wav","./Samples/Bengali_2.wav"]
)
demo3 = gr.Interface(transcribe_lang,
                     inputs = ["microphone",gr.Radio([("Hindi","hin"),("Bengali","ben"),("Odia","ory"),("Gujarati","guj"),("Telugu","tel"),("Tamil","tam"),("Marathi","mar"),("English","eng")],value="hindi")],
                   # gr.Audio(sources=["microphone"]),
                   outputs=["text","text"],
                    examples=[["./Samples/Hindi_1.mp3","hin"],["./Samples/Hindi_2.mp3","hin"],["./Samples/Hindi_3.mp3","hin"],["./Samples/Hindi_4.mp3","hin"],["./Samples/Hindi_5.mp3","hin"],["./Samples/Tamil_1.mp3","tam"],["./Samples/Tamil_2.mp3","tam"],["./Samples/Marathi_1.mp3","mar"],["./Samples/Marathi_2.mp3","mar"],["./Samples/Telugu_1.wav","tel"],["./Samples/Telugu_2.wav","tel"],["./Samples/Malayalam_1.wav","mal"],["./Samples/Malayalam_2.wav","mal"],["./Samples/Gujarati_1.wav","guj"],["./Samples/Gujarati_2.wav","guj"],["./Samples/Bengali_1.wav","ben"],["./Samples/Bengali_2.wav","ben"],["./Samples/climate ex short.wav","eng"],["./Samples/emp2.wav","eng"]]
)
                    

tabbed_interface = gr.TabbedInterface([demo,demo2,demo3],["Transcribe by auto detecting language","Detect language","Transcribe by providing language"])

with gr.Blocks() as asr:
    tabbed_interface.render()
asr.launch()
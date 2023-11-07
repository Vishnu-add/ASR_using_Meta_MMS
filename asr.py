from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import time
import gradio as gr
import librosa

model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

model_id_lid = "facebook/mms-lid-126"
processor_lid = AutoFeatureExtractor.from_pretrained(model_id_lid)
model_lid = Wav2Vec2ForSequenceClassification.from_pretrained(model_id_lid)

def transcribe(audio):
    audio = librosa.load(audio, sr=16_000, mono=True)[0]
    inputs = processor(audio, sampling_rate=16_000,return_tensors="pt")
    with torch.no_grad():
        tr_start_time = time.time()
        outputs = model(**inputs).logits
        tr_end_time = time.time()
    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)
    return transcription,(tr_end_time-tr_start_time)


def detect_language(audio):  
    audio = librosa.load(audio, sr=16_000, mono=True)[0]
    # print(audio)
    inputs_lid = processor_lid(audio, sampling_rate=16_000, return_tensors="pt")
    with torch.no_grad():
        start_time_lid = time.time()
        outputs_lid = model_lid(**inputs_lid).logits
        end_time = time.time()
#     print(end_time-start_time," sec")
    lang_id = torch.argmax(outputs_lid, dim=-1)[0].item()
    detected_lang = model_lid.config.id2label[lang_id]
    print(detected_lang)
    return detected_lang, (end_time_lid-start_time_lid)


def transcribe_lang(audio,lang):
    audio = librosa.load(audio, sr=16_000, mono=True)[0]
    processor.tokenizer.set_target_lang(lang)
    model.load_adapter(lang)
    print(lang)
    inputs = processor(audio, sampling_rate=16_000,return_tensors="pt")
    with torch.no_grad():
        tr_start_time = time.time()
        outputs = model(**inputs).logits
        tr_end_time = time.time()
    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)
    return transcription,(tr_end_time-tr_start_time)

import os
import whisper

def transcribe_audio_files(audio_dir):
    model = whisper.load_model("base")  
    transcripts = {}

    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(audio_dir, file)
            result = model.transcribe(file_path)
            transcripts[file.replace('.wav', '')] = result['text']

    return transcripts

import gc 
import json
import os
import tempfile

import dotenv
import easyocr
from moviepy import VideoFileClip
import numpy as np
from pyannote.audio import Pipeline
import pandas as pd
import torch
import tqdm
import whisperx
from whisperx.audio import SAMPLE_RATE

class Transcription:
    def __init__(self, hf_token):
        device = "cuda" 
        compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
        self.model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        self.diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(torch.device(device))

    def process_video_file(self, video_path, batch_size=16):
        with tempfile.NamedTemporaryFile(suffix='.mp3') as temp_audio_file:
            temp_audio_path = temp_audio_file.name
            # Use moviepy to extract audio
            with VideoFileClip(video_path) as video:
                video.audio.write_audiofile(temp_audio_path)
            
            audio = whisperx.load_audio(temp_audio_path)
        return self.process_audio(audio, batch_size=batch_size)

    def process_video_bytes(self, video_bytes, batch_size=16):
        # Create a temporary file to write the mp4 bytes
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video_file:
            with tempfile.NamedTemporaryFile(suffix='.mp3') as temp_audio_file:
                temp_video_path = temp_video_file.name
                temp_video_file.write(video_bytes)

                temp_audio_path = temp_audio_file.name

                # Use moviepy to extract audio
                with VideoFileClip(temp_video_path) as video:
                    video.audio.write_audiofile(temp_audio_path)
                
                audio = whisperx.load_audio(temp_audio_path)
        
        return self.process_audio(audio, batch_size=batch_size)

    def process_audio(self, audio, batch_size=16):
        device = "cuda" 

        # 1. Transcribe with original whisper (batched)
        # save model to local path (optional)
        # model_dir = "/path/"
        # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

        
        result = self.model.transcribe(audio, batch_size=batch_size)

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model

        # 2. Align whisper output
        language = result['language']
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

        # 3. Assign speaker labels
        # add min/max number of speakers if known
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments, embeddings = self.diarize_model(audio_data, return_embeddings=True)
        diarize_segments = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_segments['start'] = diarize_segments['segment'].apply(lambda x: x.start)
        diarize_segments['end'] = diarize_segments['segment'].apply(lambda x: x.end)
        
        result = whisperx.assign_word_speakers(diarize_segments, result)
        result['language'] = language
        return result, diarize_segments, embeddings

class OCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en', 'fr'], gpu=True)

    def read_video(self, video_bytes):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video:
            temp_video.write(video_bytes)
            temp_video_path = temp_video.name
            with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_image:
                temp_img_path = temp_image.name
                # extract first frame
                with VideoFileClip(temp_video_path) as video:
                    video.save_frame(temp_img_path, t=0)

                result = self.reader.readtext(temp_img_path)

        result = [r for r in result if r[2] > 0.4]
        text_lines = [x[1] for x in result]

        return text_lines

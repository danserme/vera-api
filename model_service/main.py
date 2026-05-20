import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from faster_whisper import WhisperModel

app = FastAPI(title="Whisper Model Worker Service")

# Read settings from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "./model-2")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")

model = None

@app.on_event("startup")
def load_model():
    global model
    print(f"Loading Whisper model from {MODEL_PATH} on {DEVICE} ({COMPUTE_TYPE})...")
    try:
        model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load Whisper model: {str(e)}")
        # If CUDA is requested but fails/unavailable, fallback to CPU
        if DEVICE == "cuda":
            print("Attempting fallback to CPU with int8...")
            try:
                model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
                print("Whisper model loaded successfully on CPU fallback.")
            except Exception as ex:
                print(f"CPU fallback also failed: {str(ex)}")
                raise ex
        else:
            raise e

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        print(f"Worker received request for file: {file.filename}")
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Transcribe using loaded Whisper model
        segments, info = model.transcribe(audio_buffer, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        audio_buffer.close()

        print(f"Worker transcription completed: {transcription[:100]}...")
        return {"transcription": transcription}
    
    except Exception as e:
        print(f"Worker transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed inside worker: {str(e)}")

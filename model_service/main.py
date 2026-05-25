import io
import os
import threading
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from faster_whisper import WhisperModel
from typing import Optional

app = FastAPI(title="Unified Whisper Model Worker Service")

# Read settings from environment variables
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
MODELS_DIR = os.getenv("MODELS_DIR", "/mnt/models")
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "model-1,model-2,model-b")

# Thread-safe cache of loaded WhisperModel instances
models_cache = {}
cache_lock = threading.Lock()

def get_model(model_name: str) -> WhisperModel:
    """
    Retrieves the WhisperModel from the memory cache, or loads it dynamically on GPU (with CPU fallback).
    """
    with cache_lock:
        if model_name in models_cache:
            return models_cache[model_name]
        
        # Resolve the directory path for the requested model
        # 1. Check GCSFuse mount directory first
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            # 2. Check local workspace directory for development/Docker Compose fallback
            model_path = os.path.join("/app", model_name)
            if not os.path.exists(model_path):
                # 3. Check current working directory
                model_path = os.path.join("./", model_name)
                if not os.path.exists(model_path):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model directory '{model_name}' not found under {MODELS_DIR}, /app, or current folder."
                    )
        
        print(f"Loading Whisper model '{model_name}' from {model_path} on {DEVICE} ({COMPUTE_TYPE})...", flush=True)
        try:
            loaded_model = WhisperModel(model_path, device=DEVICE, compute_type=COMPUTE_TYPE)
            models_cache[model_name] = loaded_model
            print(f"Whisper model '{model_name}' loaded successfully on GPU.", flush=True)
            return loaded_model
        except Exception as e:
            print(f"FAILED to load Whisper model '{model_name}' on GPU: {str(e)}", flush=True)
            if DEVICE == "cuda":
                print(f"Attempting CPU fallback with int8 for '{model_name}'...", flush=True)
                try:
                    loaded_model = WhisperModel(model_path, device="cpu", compute_type="int8")
                    models_cache[model_name] = loaded_model
                    print(f"Whisper model '{model_name}' loaded successfully on CPU fallback.", flush=True)
                    return loaded_model
                except Exception as ex:
                    print(f"CPU fallback for '{model_name}' also failed: {str(ex)}", flush=True)
                    raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {str(ex)}")
            else:
                raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {str(e)}")

@app.on_event("startup")
def preload_models():
    if PRELOAD_MODELS:
        for m in PRELOAD_MODELS.split(","):
            m = m.strip()
            if m:
                print(f"Preloading model '{m}' during startup...", flush=True)
                try:
                    get_model(m)
                except Exception as e:
                    print(f"Warning: Failed to preload model '{m}': {e}", flush=True)

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Query("model-b")  # Accept model name as a query parameter (defaults to model-b)
):
    try:
        print(f"Worker received request for model '{model}' with file: {file.filename}", flush=True)
        
        # Resolve/load model from cache
        whisper_model = get_model(model)
        
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Transcribe using resolved model
        segments, info = whisper_model.transcribe(audio_buffer, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        audio_buffer.close()

        print(f"Worker transcription completed for model '{model}': {transcription[:100]}...", flush=True)
        return {"transcription": transcription}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Worker transcription failed for model '{model}': {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed inside worker: {str(e)}")

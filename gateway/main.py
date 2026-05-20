from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from typing import Optional
import httpx
import os
from urllib.parse import urlparse

app = FastAPI(title="Vera API Gateway / Router")

# Define the supported models and their microservice internal URLs
# In Docker Compose, service names resolve as hostnames.
MODEL_SERVICES = {
    "model-1": os.getenv("MODEL_1_URL", "http://model-1-service:8000/transcribe"),
    "model-2": os.getenv("MODEL_2_URL", "http://model-2-service:8000/transcribe"),
    "model-b": os.getenv("MODEL_B_URL", "http://model-b-service:8000/transcribe"),
}

# The user explicitly selected 'model-b' as the default model
DEFAULT_MODEL = "model-b"

async def get_oidc_token(audience: str) -> Optional[str]:
    """
    Attempts to fetch a Google OIDC ID token from the GCP metadata server
    to authenticate with downstream private Cloud Run worker services.
    Returns None if running locally or if fetching fails.
    """
    try:
        # Parse target audience base URL (excluding paths/query parameters)
        parsed = urlparse(audience)
        audience_base = f"{parsed.scheme}://{parsed.netloc}"
        print(f"Fetching OIDC token from metadata server for audience: {audience_base}", flush=True)
        
        async with httpx.AsyncClient() as client:
            headers = {"Metadata-Flavor": "Google"}
            url = f"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={audience_base}"
            response = await client.get(url, headers=headers, timeout=2.0)
            print(f"Metadata server response status: {response.status_code}", flush=True)
            if response.status_code == 200:
                token = response.text.strip()
                print(f"Successfully retrieved OIDC token (length={len(token)})", flush=True)
                return token
            else:
                print(f"Failed to fetch OIDC token: {response.status_code} - {response.text}", flush=True)
    except Exception as e:
        print(f"Could not retrieve OIDC token (normal if running locally): {e}", flush=True)
    return None

@app.post("/transcribe")
async def transcribe(
    file: Optional[UploadFile] = File(None),
    wav: Optional[UploadFile] = File(None),
    model: Optional[str] = Query(None),
    model_form: Optional[str] = Form(None, alias="model")
):
    # Resolve the uploaded file from either 'file' or 'wav' field
    audio_file = file or wav
    if not audio_file:
        raise HTTPException(
            status_code=400, 
            detail="No audio file uploaded. Please upload a file using the 'file' or 'wav' field."
        )

    # Resolve model selection from query parameters or form parameters
    selected_model = model or model_form or DEFAULT_MODEL
    
    if selected_model not in MODEL_SERVICES:
        supported = ", ".join(MODEL_SERVICES.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{selected_model}'. Supported models: {supported}"
        )

    target_url = MODEL_SERVICES[selected_model]
    print(f"Routing transcription request to {selected_model} at {target_url}...", flush=True)

    try:
        # Read the uploaded file bytes
        file_bytes = await audio_file.read()
        
        # Check if we need to obtain OIDC token for secure downstream Cloud Run request
        headers = {}
        if target_url.startswith("https://"):
            token = await get_oidc_token(target_url)
            if token:
                headers["Authorization"] = f"Bearer {token}"
                print(f"Acquired OIDC token for downstream {selected_model} authentication", flush=True)
            else:
                print(f"WARNING: No OIDC token acquired for downstream {selected_model} request", flush=True)
        
        # Asynchronously forward the file to the downstream microservice
        # We specify a generous timeout (e.g. 60 seconds) for large audio files
        async with httpx.AsyncClient() as client:
            files = {"file": (audio_file.filename, file_bytes, audio_file.content_type)}
            response = await client.post(target_url, files=files, headers=headers, timeout=60.0)

        # Handle downstream failures
        if response.status_code != 200:
            print(f"Downstream service {selected_model} failed with status {response.status_code}: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Downstream transcription failed: {response.text}"
            )

        result = response.json()
        transcription = result.get("transcription", "")

        print(f"Successfully received transcription from {selected_model}: {transcription[:100]}...")

        # Return both keys for complete backwards compatibility
        # - "transcript" for the Flutter client
        # - "transcription" for standard REST API users
        return {
            "transcript": transcription,
            "transcription": transcription,
            "model_used": selected_model
        }

    except httpx.RequestError as exc:
        print(f"Gateway request error connecting to {selected_model}: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"Downstream transcription service '{selected_model}' is currently unreachable."
        )
    except Exception as e:
        print(f"Gateway unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Gateway routing error: {str(e)}"
        )

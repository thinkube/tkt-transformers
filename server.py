# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: MIT

"""Serves one Hugging Face model with the OpenAI route of its task.

The LLM gateway creates the pod with the model's id (MODEL_ID) and the
catalog's task (TASK). The model was mirrored from Hugging Face into MLflow;
its files are under /mlflow-models, and it is loaded from there once, before
the port opens.

    automatic-speech-recognition  POST /v1/audio/transcriptions  transformers
    text-to-image                 POST /v1/images/generations    diffusers
    text-to-speech                POST /v1/audio/speech          transformers

Every pod also answers GET /v1/models and GET /health.
"""

import base64
import io
import os
import sys
import time
from contextlib import asynccontextmanager

import httpx
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

MLFLOW_URL = "http://mlflow.mlflow.svc.cluster.local:5000"
TASKS = ("automatic-speech-recognition", "text-to-image", "text-to-speech")


def required(name: str, why: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set: {why}")
    return value


MODEL_ID = required("MODEL_ID", "the LLM gateway sets it when it creates the pod")
TASK = required("TASK", "the LLM gateway sets it from the model's catalog task")
if TASK not in TASKS:
    sys.exit(f"{MODEL_ID} has the task '{TASK}', which this server does not serve; it serves {', '.join(TASKS)}")


def model_dir() -> str:
    """The folder of the latest MLflow version of MODEL_ID."""
    names = ("MLFLOW_KEYCLOAK_TOKEN_URL", "MLFLOW_KEYCLOAK_CLIENT_ID", "MLFLOW_CLIENT_SECRET",
             "MLFLOW_AUTH_USERNAME", "MLFLOW_AUTH_PASSWORD")
    env = {n: required(n, "the component's MLflow credentials Secret provides it") for n in names}
    resp = httpx.post(env["MLFLOW_KEYCLOAK_TOKEN_URL"], data={
        "grant_type": "password", "client_id": env["MLFLOW_KEYCLOAK_CLIENT_ID"],
        "client_secret": env["MLFLOW_CLIENT_SECRET"], "username": env["MLFLOW_AUTH_USERNAME"],
        "password": env["MLFLOW_AUTH_PASSWORD"], "scope": "openid",
    }, timeout=30)
    if resp.status_code != 200:
        sys.exit(f"Keycloak refused the MLflow token: HTTP {resp.status_code}: {resp.text}")
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

    registered = MODEL_ID.replace("/", "-")
    resp = httpx.get(f"{MLFLOW_URL}/api/2.0/mlflow/model-versions/search",
                     params={"filter": f"name='{registered}'"}, headers=headers, timeout=30)
    resp.raise_for_status()
    versions = resp.json().get("model_versions", [])
    if not versions:
        sys.exit(f"{MODEL_ID} is not registered in MLflow as {registered}; mirror it first")
    run_id = max(versions, key=lambda v: int(v["version"]))["run_id"]
    resp = httpx.get(f"{MLFLOW_URL}/api/2.0/mlflow/runs/get", params={"run_id": run_id},
                     headers=headers, timeout=30)
    resp.raise_for_status()
    experiment_id = resp.json()["run"]["info"]["experiment_id"]
    return f"/mlflow-models/artifacts/{experiment_id}/{run_id}/artifacts/model"


def load(path: str):
    if not torch.cuda.is_available():
        sys.exit("No CUDA device is visible to this pod")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if TASK == "text-to-image":
        # The pipeline class is the one the model's model_index.json names.
        from diffusers import DiffusionPipeline
        return DiffusionPipeline.from_pretrained(path, torch_dtype=dtype).to("cuda")
    from transformers import pipeline
    return pipeline(TASK, model=path, torch_dtype=dtype, device="cuda")


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    path = model_dir()
    print(f"Loading {MODEL_ID} ({TASK}) from {path}", flush=True)
    start = time.time()
    model = load(path)
    print(f"Loaded in {time.time() - start:.0f}s", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


def check_model(name: str):
    if name != MODEL_ID:
        raise HTTPException(404, f"This server serves {MODEL_ID}, not {name}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "owned_by": "thinkube", "task": TASK}]}


if TASK == "automatic-speech-recognition":
    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model_name: str = Form(..., alias="model"),
        language: str | None = Form(None),
        response_format: str = Form("json"),
    ):
        check_model(model_name)
        if response_format not in ("json", "text"):
            raise HTTPException(400, f"response_format '{response_format}' is not served; use json or text")
        audio = await file.read()
        kwargs = {"generate_kwargs": {"language": language}} if language else {}
        # Audio longer than 30 seconds is transcribed in 30-second windows.
        result = model(audio, chunk_length_s=30, **kwargs)
        text = result["text"].strip()
        if response_format == "text":
            return PlainTextResponse(text)
        return {"text": text}


if TASK == "text-to-image":
    class ImageRequest(BaseModel):
        model: str
        prompt: str
        n: int = 1
        size: str | None = None
        response_format: str = "b64_json"
        num_inference_steps: int | None = None
        guidance_scale: float | None = None

    @app.post("/v1/images/generations")
    def images(req: ImageRequest):
        check_model(req.model)
        if req.response_format != "b64_json":
            raise HTTPException(400, "Only response_format b64_json is served; this server keeps no image files")
        kwargs = {"num_images_per_prompt": req.n}
        if req.size:
            try:
                width, height = (int(v) for v in req.size.lower().split("x"))
            except ValueError:
                raise HTTPException(400, f"size '{req.size}' is not WIDTHxHEIGHT")
            kwargs.update(width=width, height=height)
        if req.num_inference_steps is not None:
            kwargs["num_inference_steps"] = req.num_inference_steps
        if req.guidance_scale is not None:
            kwargs["guidance_scale"] = req.guidance_scale
        images = model(req.prompt, **kwargs).images
        data = []
        for image in images:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
        return JSONResponse({"created": int(time.time()), "data": data})


if TASK == "text-to-speech":
    class SpeechRequest(BaseModel):
        model: str
        input: str
        voice: str | None = None
        response_format: str = "wav"

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest):
        import soundfile
        check_model(req.model)
        if req.response_format != "wav":
            raise HTTPException(400, f"response_format '{req.response_format}' is not served; use wav")
        if req.voice not in (None, "default"):
            raise HTTPException(400, f"voice '{req.voice}' is not served: the transformers text-to-speech "
                                     "pipeline speaks with the model's own voice; send voice 'default'")
        out = model(req.input)
        buf = io.BytesIO()
        soundfile.write(buf, out["audio"].squeeze(), out["sampling_rate"], format="WAV")
        return Response(buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

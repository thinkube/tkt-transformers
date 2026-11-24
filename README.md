# tkt-transformers

A server for Hugging Face models that are not language models, behind the
LLM Gateway: speech to text, text to image, text to speech, and trained
parts on a base model (taggers). It is the
`transformers` optional component of Thinkube.

## What it does

The container runs `server.py` (FastAPI) on port 8000. It loads one model,
the one the gateway names in `MODEL_ID`, and serves the OpenAI route of the
model's catalog task (`TASK`):

| Task | Route | Library |
|---|---|---|
| `automatic-speech-recognition` | `POST /v1/audio/transcriptions` | transformers pipeline |
| `text-to-image` | `POST /v1/images/generations` | diffusers, the pipeline the model's `model_index.json` names |
| `text-to-speech` | `POST /v1/audio/speech` | transformers pipeline |
| `heads` | `POST /v1/heads` | meepserve: a trained part (a tagger, a choice head) on its base model |

Every pod also answers `GET /v1/models` and `GET /health`. A model with
another task stops the pod with an error that names the tasks served.

- **The model comes from MLflow.** The server finds the latest MLflow
  version of `MODEL_ID` and loads it from `/mlflow-models` before it opens
  its port.
- **Speech to text** takes the multipart form of the OpenAI API: `file`,
  `model`, and optionally `language` and `response_format` (`json` or
  `text`). Audio longer than 30 seconds is transcribed in 30-second windows.
  ffmpeg decodes the file.
- **Text to image** takes `prompt`, `n` and `size` (`WIDTHxHEIGHT`), and
  `num_inference_steps` and `guidance_scale` beyond the OpenAI fields. The
  images come back as base64 PNG (`b64_json`); no image files are kept.
- **Heads** serves a trained part on a base model: the model's folder has a
  `meepserve.json` naming the part's kind and its base, and the base is
  loaded from its own MLflow mirror. The body names the model, and the rest
  of it is the part's kind's request; a request the part cannot answer gets
  HTTP 400 with the reason. Requests are answered one at a time.
- **Text to speech** takes `input` and answers with a WAV file. The pipeline
  speaks with the model's own voice: `voice` is `default` or left out.

This is a plain server: one request at a time, no batching. It suits one
user and models no dedicated engine serves.

## How it reaches a user

It is installed and removed from the Optional Components page in
thinkube-control, with no pods until a model is loaded. `spec.llm_backend`
in `thinkube.yaml` declares it a `performance` backend of the LLM Gateway of
type `transformers`. A model runs on it when its catalog entry lists
`transformers` in `server_type` and has one of the tasks above.

## Image

The container is built on Thinkube's `ai-inference-base` image (CUDA 13.0,
PyTorch 2.9, amd64 and arm64), with transformers, diffusers and FastAPI from
`requirements.txt`, and ffmpeg and libsndfile.

## Licence

This template is MIT licensed. Each model has its own licence, stated in its
model card.

import json
from typing import Any, Dict, List, Optional, Union
import aiohttp
from dotenv import load_dotenv
from httpx import AsyncClient

from ui.routers import chats
from config import OPENAI_API_BASE_URLS, OPENAI_API_KEYS, AppConfig
from misc import add_or_update_system_message, get_last_user_message
from utils import get_rag_context, rag_template

load_dotenv()

import logging
import os

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

from rag import app as rag_app
# from app.gradio_ui.ui import gradio_iface

# from instrument import instrument


app = FastAPI(title="RAG x FastAPI")


environment = os.getenv("ENVIRONMENT", "dev")  # Default to 'development' if not set

app.state.config = AppConfig()
app.state.config.OPENAI_API_BASE_URLS = OPENAI_API_BASE_URLS
app.state.config.OPENAI_API_KEYS = OPENAI_API_KEYS

log = logging.getLogger("uvicorn")

if environment == "dev":
    log.warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/api/version")
async def get_app_config():

    return {
        "version": "demo",
    }


@app.post("/chat/completions")
@app.post("/chat/completions/{url_idx}")
async def generate_chat_completion(
    form_data: dict,
    url_idx: Optional[int] = None,
):
    idx = 0
    payload = {**form_data}

    if "chat_id" in payload:
        del payload["chat_id"]

    if "title" in payload:
        del payload["title"]

    if "task" in payload:
        del payload["task"]

    # Convert the modified body back to JSON
    payload = json.dumps(payload)

    log.debug(payload)

    url = app.state.config.OPENAI_API_BASE_URLS[idx]
    key = app.state.config.OPENAI_API_KEYS[idx]

    headers = {}
    headers["Authorization"] = f"Bearer {key}"
    headers["Content-Type"] = "application/json"

    r = None
    session = None
    streaming = False

    try:
        session = aiohttp.ClientSession(trust_env=True)
        r = await session.request(
            method="POST",
            url=f"{url}/chat/completions",
            data=payload,
            headers=headers,
        )

        r.raise_for_status()

        # Check if response is SSE
        if "text/event-stream" in r.headers.get("Content-Type", ""):
            streaming = True
            return StreamingResponse(
                r.content,
                status_code=r.status,
                headers=dict(r.headers),
                background=BackgroundTask(
                    cleanup_response, response=r, session=session
                ),
            )
        else:
            response_data = await r.json()
            return response_data
    except Exception as e:
        log.exception(e)
        error_detail = "Server Connection Error"
        if r is not None:
            try:
                res = await r.json()
                print(res)
                if "error" in res:
                    error_detail = f"External: {res['error']['message'] if 'message' in res['error'] else res['error']}"
            except:
                error_detail = f"External: {e}"
        raise HTTPException(status_code=r.status if r else 500, detail=error_detail)
    finally:
        if not streaming and session:
            if r:
                r.close()
            await session.close()

async def cleanup_response(
    response: Optional[aiohttp.ClientResponse],
    session: Optional[aiohttp.ClientSession],
):
    if response:
        response.close()
    if session:
        await session.close()

@app.post("/chat/demo_rag")
async def generate_chat_completion(
    form_data: dict
):
    val = form_data['model']
    return {"value": val }

class ChatCompletionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        data_items = []

        show_citations = False
        citations = []

        if request.method == "POST" and any(
            endpoint in request.url.path
            for endpoint in ["/chat/demo_rag", "/chat/completions"]
        ):
            log.debug(f"request.url.path: {request.url.path}")

            # Read the original request body
            body = await request.body()
            body_str = body.decode("utf-8")
            data = json.loads(body_str) if body_str else {}

            # Flag to skip RAG completions if file_handler is present in tools/functions
            skip_files = False

            prompt = get_last_user_message(data["messages"])
            context = ""

            # If files field is present, generate RAG completions
            # If skip_files is True, skip the RAG completions
            if "files" in data:
                if not skip_files:
                    data = {**data}
                    rag_context, rag_citations = get_rag_context(
                        files=data["files"],
                        messages=data["messages"],
                        k=rag_app.state.config.TOP_K,
                        embedding_function=rag_app.state.EMBEDDING_FUNCTION,
                        reranking_function=None,
                        r=None,
                        hybrid_search=rag_app.state.config.ENABLE_RAG_HYBRID_SEARCH,
                    )
                    if rag_context:
                        context += ("\n" if context != "" else "") + rag_context

                    log.debug(f"rag_context: {rag_context}, citations: {citations}")

                del data["files"]

            if context != "":
                system_prompt = rag_template(
                    rag_app.state.config.RAG_TEMPLATE, context, prompt
                )
                print(system_prompt)
                data["messages"] = add_or_update_system_message(
                    system_prompt, data["messages"]
                )

            modified_body_bytes = json.dumps(data).encode("utf-8")
            # Replace the request body with the modified one
            request._body = modified_body_bytes
            # Set custom header to ensure content-length matches new body length
            request.headers.__dict__["_list"] = [
                (b"content-length", str(len(modified_body_bytes)).encode("utf-8")),
                *[
                    (k, v)
                    for k, v in request.headers.raw
                    if k.lower() != b"content-length"
                ],
            ]

            response = await call_next(request)
            if isinstance(response, StreamingResponse):
                # If it's a streaming response, inject it as SSE event or NDJSON line
                content_type = response.headers.get("Content-Type")
                if "text/event-stream" in content_type:
                    return StreamingResponse(
                        self.openai_stream_wrapper(response.body_iterator, data_items),
                    )

                return response
            else:
                return response
        # If it's not a chat completion request, just pass it through
        response = await call_next(request)
        return response
    
    async def _receive(self, body: bytes):
        return {"type": "http.request", "body": body, "more_body": False}

    async def openai_stream_wrapper(self, original_generator, data_items):
        for item in data_items:
            yield f"data: {json.dumps(item)}\n\n"

        async for data in original_generator:
            yield data


app.add_middleware(ChatCompletionMiddleware)

app.include_router(chats.router, prefix="/chats", tags=["chats"])
app.mount( "/api/rag", rag_app)



if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=7000, reload=True)
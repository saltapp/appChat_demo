import json
from typing import Optional
from dotenv import load_dotenv

from misc import add_or_update_system_message, get_last_user_message
from utils import get_rag_context, rag_template

load_dotenv()

import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from rag import app as rag_app
# from app.gradio_ui.ui import gradio_iface

# from instrument import instrument


app = FastAPI(title="RAG x FastAPI")


environment = os.getenv("ENVIRONMENT", "dev")  # Default to 'development' if not set

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

# @app.post("/chat/completions")
# @app.post("/chat/completions/{url_idx}")
# async def generate_chat_completion(
#     form_data: dict,
#     url_idx: Optional[int] = None,
# ):
#     idx = 0
#     payload = {**form_data}

#     model_id = form_data.get("model")
#     model_info = Models.get_model_by_id(model_id)

#     if model_info:
#         if model_info.base_model_id:
#             payload["model"] = model_info.base_model_id

#         model_info.params = model_info.params.model_dump()

#         if model_info.params:
#             if model_info.params.get("temperature", None) is not None:
#                 payload["temperature"] = float(model_info.params.get("temperature"))

#             if model_info.params.get("top_p", None):
#                 payload["top_p"] = int(model_info.params.get("top_p", None))

#             if model_info.params.get("max_tokens", None):
#                 payload["max_tokens"] = int(model_info.params.get("max_tokens", None))

#             if model_info.params.get("frequency_penalty", None):
#                 payload["frequency_penalty"] = int(
#                     model_info.params.get("frequency_penalty", None)
#                 )

#             if model_info.params.get("seed", None):
#                 payload["seed"] = model_info.params.get("seed", None)

#             if model_info.params.get("stop", None):
#                 payload["stop"] = (
#                     [
#                         bytes(stop, "utf-8").decode("unicode_escape")
#                         for stop in model_info.params["stop"]
#                     ]
#                     if model_info.params.get("stop", None)
#                     else None
#                 )

#         system = model_info.params.get("system", None)
#         if system:
#             system = prompt_template(
#                 system,
#                 **(
#                     {
#                         "user_name": user.name,
#                         "user_location": (
#                             user.info.get("location") if user.info else None
#                         ),
#                     }
#                     if user
#                     else {}
#                 ),
#             )
#             # Check if the payload already has a system message
#             # If not, add a system message to the payload
#             if payload.get("messages"):
#                 for message in payload["messages"]:
#                     if message.get("role") == "system":
#                         message["content"] = system + message["content"]
#                         break
#                 else:
#                     payload["messages"].insert(
#                         0,
#                         {
#                             "role": "system",
#                             "content": system,
#                         },
#                     )

#     else:
#         pass

#     model = app.state.MODELS[payload.get("model")]
#     idx = model["urlIdx"]

#     if "pipeline" in model and model.get("pipeline"):
#         payload["user"] = {
#             "name": user.name,
#             "id": user.id,
#             "email": user.email,
#             "role": user.role,
#         }

#     # Check if the model is "gpt-4-vision-preview" and set "max_tokens" to 4000
#     # This is a workaround until OpenAI fixes the issue with this model
#     if payload.get("model") == "gpt-4-vision-preview":
#         if "max_tokens" not in payload:
#             payload["max_tokens"] = 4000
#         log.debug("Modified payload:", payload)

#     # Convert the modified body back to JSON
#     payload = json.dumps(payload)

#     log.debug(payload)

#     url = app.state.config.OPENAI_API_BASE_URLS[idx]
#     key = app.state.config.OPENAI_API_KEYS[idx]

#     headers = {}
#     headers["Authorization"] = f"Bearer {key}"
#     headers["Content-Type"] = "application/json"

#     r = None
#     session = None
#     streaming = False

#     try:
#         session = aiohttp.ClientSession(trust_env=True)
#         r = await session.request(
#             method="POST",
#             url=f"{url}/chat/completions",
#             data=payload,
#             headers=headers,
#         )

#         r.raise_for_status()

#         # Check if response is SSE
#         if "text/event-stream" in r.headers.get("Content-Type", ""):
#             streaming = True
#             return StreamingResponse(
#                 r.content,
#                 status_code=r.status,
#                 headers=dict(r.headers),
#                 background=BackgroundTask(
#                     cleanup_response, response=r, session=session
#                 ),
#             )
#         else:
#             response_data = await r.json()
#             return response_data
#     except Exception as e:
#         log.exception(e)
#         error_detail = "Open WebUI: Server Connection Error"
#         if r is not None:
#             try:
#                 res = await r.json()
#                 print(res)
#                 if "error" in res:
#                     error_detail = f"External: {res['error']['message'] if 'message' in res['error'] else res['error']}"
#             except:
#                 error_detail = f"External: {e}"
#         raise HTTPException(status_code=r.status if r else 500, detail=error_detail)
#     finally:
#         if not streaming and session:
#             if r:
#                 r.close()
#             await session.close()

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
            for endpoint in ["/chat/demo_rag"]
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
    
    async def _receive(self, body: bytes):
        return {"type": "http.request", "body": body, "more_body": False}

    async def openai_stream_wrapper(self, original_generator, data_items):
        for item in data_items:
            yield f"data: {json.dumps(item)}\n\n"

        async for data in original_generator:
            yield data


app.add_middleware(ChatCompletionMiddleware)
app.mount( "/api/rag", rag_app)



if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", reload=True)
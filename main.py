from dotenv import load_dotenv

load_dotenv()

import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from rag import app as ragapp
# from app.gradio_ui.ui import gradio_iface

# from instrument import instrument


app = FastAPI(title="RAG x FastAPI")


environment = os.getenv("ENVIRONMENT", "dev")  # Default to 'development' if not set


if environment == "dev":
    logger = logging.getLogger("uvicorn")
    logger.warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/api/version")
async def get_app_config():
    breakpoint()
    return {
        "version": "demo",
    }

@app.post("/api/chat/completions")
async def generate_chat_completions(form_data: dict, user=Depends(get_verified_user)):
    model_id = form_data["model"]
    if model_id not in app.state.MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    return await generate_openai_chat_completion(form_data, user=user)

@app.post("/api/chat/completed")
async def chat_completed(form_data: dict, user=Depends(get_verified_user)):
    data = form_data
    model_id = data["model"]
    if model_id not in app.state.MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    model = app.state.MODELS[model_id]

    filters = [
        model
        for model in app.state.MODELS.values()
        if "pipeline" in model
        and "type" in model["pipeline"]
        and model["pipeline"]["type"] == "filter"
        and (
            model["pipeline"]["pipelines"] == ["*"]
            or any(
                model_id == target_model_id
                for target_model_id in model["pipeline"]["pipelines"]
            )
        )
    ]

    sorted_filters = sorted(filters, key=lambda x: x["pipeline"]["priority"])
    if "pipeline" in model:
        sorted_filters = [model] + sorted_filters

    for filter in sorted_filters:
        r = None
        try:
            urlIdx = filter["urlIdx"]

            url = openai_app.state.config.OPENAI_API_BASE_URLS[urlIdx]
            key = openai_app.state.config.OPENAI_API_KEYS[urlIdx]

            if key != "":
                headers = {"Authorization": f"Bearer {key}"}
                r = requests.post(
                    f"{url}/{filter['id']}/filter/outlet",
                    headers=headers,
                    json={
                        "user": {
                            "id": user.id,
                            "name": user.name,
                            "email": user.email,
                            "role": user.role,
                        },
                        "body": data,
                    },
                )

                r.raise_for_status()
                data = r.json()
        except Exception as e:
            # Handle connection error here
            print(f"Connection error: {e}")

            if r is not None:
                try:
                    res = r.json()
                    if "detail" in res:
                        return JSONResponse(
                            status_code=r.status_code,
                            content=res,
                        )
                except:
                    pass

            else:
                pass

    def get_priority(function_id):
        function = Functions.get_function_by_id(function_id)
        if function is not None and hasattr(function, "valves"):
            return (function.valves if function.valves else {}).get("priority", 0)
        return 0

    filter_ids = [function.id for function in Functions.get_global_filter_functions()]
    if "info" in model and "meta" in model["info"]:
        filter_ids.extend(model["info"]["meta"].get("filterIds", []))
        filter_ids = list(set(filter_ids))

    enabled_filter_ids = [
        function.id
        for function in Functions.get_functions_by_type("filter", active_only=True)
    ]
    filter_ids = [
        filter_id for filter_id in filter_ids if filter_id in enabled_filter_ids
    ]

    # Sort filter_ids by priority, using the get_priority function
    filter_ids.sort(key=get_priority)

    for filter_id in filter_ids:
        filter = Functions.get_function_by_id(filter_id)
        if filter:
            if filter_id in webui_app.state.FUNCTIONS:
                function_module = webui_app.state.FUNCTIONS[filter_id]
            else:
                function_module, function_type, frontmatter = (
                    load_function_module_by_id(filter_id)
                )
                webui_app.state.FUNCTIONS[filter_id] = function_module

            if hasattr(function_module, "valves") and hasattr(
                function_module, "Valves"
            ):
                valves = Functions.get_function_valves_by_id(filter_id)
                function_module.valves = function_module.Valves(
                    **(valves if valves else {})
                )

            try:
                if hasattr(function_module, "outlet"):
                    outlet = function_module.outlet

                    # Get the signature of the function
                    sig = inspect.signature(outlet)
                    params = {"body": data}

                    if "__user__" in sig.parameters:
                        __user__ = {
                            "id": user.id,
                            "email": user.email,
                            "name": user.name,
                            "role": user.role,
                        }

                        try:
                            if hasattr(function_module, "UserValves"):
                                __user__["valves"] = function_module.UserValves(
                                    **Functions.get_user_valves_by_id_and_user_id(
                                        filter_id, user.id
                                    )
                                )
                        except Exception as e:
                            print(e)

                        params = {**params, "__user__": __user__}

                    if "__id__" in sig.parameters:
                        params = {
                            **params,
                            "__id__": filter_id,
                        }

                    if inspect.iscoroutinefunction(outlet):
                        data = await outlet(**params)
                    else:
                        data = outlet(**params)

            except Exception as e:
                print(f"Error: {e}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": str(e)},
                )

    return data


app.mount( "/api/rag", ragapp)



if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", reload=True)
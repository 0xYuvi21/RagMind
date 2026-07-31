import base64
import time
from pathlib import Path
from typing import Optional

import requests
from langchain_core.tools import StructuredTool

# Defaults only — which Ollama endpoint/model is actually used in the running
# app is set in config.yaml (ollama_url / ollama_vlm_model) and passed in by
# Backend/conversation_service.py's ConversationService via build_read_image_tool()
# below, not read from here. These constants exist only for the CLI demo at
# the bottom of this file.
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "richardyoung/smolvlm2-2.2b-instruct:latest"


def build_read_image_tool(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    audit_logger=None,
    conversation_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> StructuredTool:
    """Builds the `read_image` tool bound to a specific Ollama endpoint/model —
    a closure, the same DI pattern Retrieve/llmquery.py's build_retrieval_tool()
    uses, so swapping the VLM model means editing config.yaml, not this file.
    `audit_logger` (Backend/interfaces.py's AuditLogger) is optional so the
    CLI demo at the bottom of this file and any other caller outside the web
    app can still build a working tool without wiring one in."""

    def _log(image_path: str, started: float, success: bool, error_message: Optional[str] = None) -> None:
        if audit_logger is None:
            return
        audit_logger.log_vlm_event(
            model=ollama_model,
            image_path=image_path,
            success=success,
            conversation_id=conversation_id,
            user_id=user_id,
            latency_ms=int((time.monotonic() - started) * 1000),
            error_message=error_message,
        )

    def _read_image(image_path: str, user_prompt: str) -> str:
        started = time.monotonic()
        try:
            path = Path(image_path)
            if not path.exists():
                error = f"Error: The image file at {image_path} does not exist."
                _log(image_path, started, success=False, error_message=error)
                return error

            with open(path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            print(f"\n[VLM] Analyzing image with Ollama ({ollama_model}): {path.name}...")

            payload = {
                "model": ollama_model,
                "prompt": user_prompt,
                "images": [encoded_image],  # Ollama expects a list of base64 strings
                "stream": False,
            }

            response = requests.post(ollama_url, json=payload)
            response.raise_for_status()

            result = response.json()
            _log(image_path, started, success=True)
            return result.get("response", "No response generated.")

        except requests.exceptions.ConnectionError:
            error = "Error: Could not connect to Ollama. Is Ollama running locally?"
            _log(image_path, started, success=False, error_message=error)
            return error
        except Exception as e:
            error = f"Error processing image with VLM: {str(e)}"
            _log(image_path, started, success=False, error_message=error)
            return error

    return StructuredTool.from_function(
        func=_read_image,
        name="read_image",
        description=(
            "Analyzes an image and answers the user's prompt based on its visual "
            "contents. Args: image_path (absolute path to the image file), "
            "user_prompt (what to ask about the image, e.g. 'Describe the diagram' "
            "or 'Read the text'). Returns the text explanation/summary from the model."
        ),
    )


if __name__ == "__main__":
    # Test the tool directly
    test_img = "C:/Users/muthi/Desktop/wonders/RAG/docs/Agentic-RAG-1.jpg"
    tool = build_read_image_tool()
    print(tool.invoke({"image_path": test_img, "user_prompt": "Describe this diagram."}))

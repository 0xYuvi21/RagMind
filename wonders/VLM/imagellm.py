import base64
import requests
from pathlib import Path
from langchain_core.tools import tool

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "richardyoung/smolvlm2-2.2b-instruct:latest"


@tool
def read_image(image_path: str, user_prompt: str) -> str:
    """
    Analyzes an image and answers the user's prompt based on its visual contents.

    Args:
        image_path (str): The absolute path to the image file to analyze (e.g., C:/path/to/image.jpg).
        user_prompt (str): What to ask the model about the image (e.g., "Describe the diagram" or "Read the text").

    Returns:
        str: The text explanation, summary, or extracted data from the image.
    """
    try:
        path = Path(image_path)
        if not path.exists():
            return f"Error: The image file at {image_path} does not exist."

        # Open and encode image to base64
        with open(path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        print(f"\n[VLM] Analyzing image with Ollama ({OLLAMA_MODEL}): {path.name}...")

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": user_prompt,
            "images": [encoded_image],  # Ollama expects a list of base64 strings
            "stream": False,
        }

        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "No response generated.")

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Is Ollama running locally?"
    except Exception as e:
        return f"Error processing image with VLM: {str(e)}"


if __name__ == "__main__":
    # Test the tool directly
    test_img = "C:/Users/muthi/Desktop/wonders/RAG/docs/Agentic-RAG-1.jpg"
    print(
        read_image.invoke(
            {"image_path": test_img, "user_prompt": "Describe this diagram."}
        )
    )

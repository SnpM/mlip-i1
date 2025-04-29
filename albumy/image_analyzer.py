from openai import OpenAI
import os

from typing import final
from pydantic import BaseModel
import base64
import json

# Initialize openai client
key = os.getenv("OPENAI_API_KEY")
CLIENT = OpenAI(api_key=key)

MODEL = "gpt-4o-mini"

@final
class ImageResult:
    alt_text: str
    tags: list[str]
    def __init__(self, alt_text: str, tags: list[str]):
        self.alt_text = alt_text
        self.tags = tags
    def __repr__(self):
        return f"ImageResult(alt_text={self.alt_text}, tags={self.tags})"
    
prompt = """
Analyze the attached image and provide alternative text and a list of tags in the following JSON format:
{
    "alt_text": "A description of the image",
    "tags": ["tag1", "tag2", "tag3"]
}

Example 1:
{
    "alt_text": "A beautiful sunset over the mountains",
    "tags": ["sunset", "mountains", "nature"]
}
Example 2:
{
    "alt_text": "A group of people sitting around a table",
    "tags": ["people", "table", "meeting"]
}
"""
class prompt_model(BaseModel):
    alt_text:str
    tags:list

def analyze_image(image_path) -> ImageResult:
    """
    Analyze an image and return the alt text and tags within an ImageResult object.
    """
    # Read and encode the image
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # Build the base64 data URL
    data_url = f"data:image/jpeg;base64,{img_base64}"

    # Send to OpenAI vision model
    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt.strip()
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ],
#        response_format="json",
        max_tokens=500
    )

    # Parse response
    content = response.choices[0].message.content

    try:
        
        parsed = json.loads(content)
        return ImageResult(alt_text=parsed["alt_text"], tags=parsed["tags"])
    except Exception as e:
        # Print error stacktrace
        print(f"Error parsing response: {e}")
        raise ValueError(f"Failed to parse the model response: {e}\nResponse was: {content}")


if __name__ == "__main__":
    # Example usage
    test_img = "/home/nibs/mlip-i1/uploads/random_0.jpg"
    response = analyze_image(test_img)
    print(response)

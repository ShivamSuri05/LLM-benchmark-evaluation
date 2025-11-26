import os
from dotenv import load_dotenv
from openai import OpenAI

class LLMClient:
    """
    Wrapper for OpenRouter-compatible OpenAI client.
    Loads API key from .env and provides a simple generate() method.
    """

    def __init__(self, model: str = "openai/gpt-4o"):
        # Load .env variables
        load_dotenv()

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Please set it in your .env file."
            )

        # Initialize OpenAI client using OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the model and return the generated text output.
        """
        try:
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Benchmark-LLM-Tool",
                },
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print("Error in LLMClient.generate():", e)
            return ""

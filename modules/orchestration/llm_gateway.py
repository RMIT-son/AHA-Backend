import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient():
    """
    A client class for interacting with the OpenAI Chat API.
    Supports dynamic configuration of model, instruction, max tokens, and temperature.
    """
    def __init__(self, model="gpt-4o-mini", instruction=None, max_tokens=1024, temperature=0.7):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = model
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.temperature = temperature

    def query(self, prompt:str):
        """
        Sends a prompt to the OpenAI Chat API using the current configuration.
        Returns the model's response text or an error message if the API call fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.instruction},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print("API error:", e)
            return "Error: No response from model."
    
    def set_model(self, model: str):
        self.model = model

    def set_instruction(self, instruction: str):
        self.instruction = instruction

    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature

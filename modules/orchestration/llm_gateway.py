import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient():
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def query(self, prompt:str, model:str, instruction:str, max_tokens:int, temperature:float):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print("API error:", e)
            return "Error: No response from model."
from openai import OpenAI
from typing import Callable

MODELS = ['llama3-8b', 'DeepSeek-14b', 'qwen14b','Qwen2.5-32b','llama3']

class ModelWrapper:
    def __init__(self, model_name, base_url, api_key):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
    
    
    def __call__(self, content) -> str:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        completion = client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
        )
        
        answer = completion.choices[0].message.content
        
        if 'deepseek' in self.model_name.lower():
            index = answer.find('</think>')
            if index != -1:
                answer = answer[index + len('</think>'):].strip()
        return [answer]


def LLM_CLS(model_name: str, base_url:str, openai_api_key: str) -> Callable:
    if model_name in MODELS:
        return ModelWrapper(model_name, base_url, openai_api_key)
    else:
        raise ValueError(f"Unknown LLM model name: {model_name}")
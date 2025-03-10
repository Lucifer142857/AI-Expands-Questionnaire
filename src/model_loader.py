import os
import openai
from llama_cpp import Llama
from . import config

class LocalModelWrapper:
    def __init__(self, model_path, n_ctx=2048):
        self.llm = Llama(
            model_path=model_path,
            n_threads=config.N_THREADS,
            n_gpu_layers=config.USE_GPU_LAYERS,
            n_ctx=n_ctx,
            n_batch=512,
            verbose=False
        )

    def create_completion(self, prompt, max_tokens=1024, temperature=0.2):
        """
        调用本地Llama模型，返回类似 OpenAI 的{"choices":[{"text": "..."}]}结构。
        """
        output = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return output


class OpenAIModelWrapper:
    def __init__(self, model_name="deepseek-reasoner"):
        # 此处已经正确初始化了客户端
        from openai import OpenAI
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
        )
        self.model_name = model_name

    def create_completion(self, prompt, max_tokens=4096, temperature=0.15):
        # 修改这里：使用客户端实例而不是全局openai模块
        response = self.client.chat.completions.create(  # 改为self.client
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        return {
            "choices": [
                {"text": response.choices[0].message.content}  # 新API的响应结构
            ]
        }

def load_model_for_4steps():
    if config.USE_OPENAI_FOR_4STEPS:
        print("  [INFO] 使用 OpenAI API 进行前半段聚类 & 画像")
        return OpenAIModelWrapper(model_name=config.OPENAI_MODEL_NAME)
    else:
        print("  [INFO] 使用本地模型 进行前半段聚类 & 画像")
        return LocalModelWrapper(model_path=config.MODEL_PATH_4STEPS, n_ctx=4096)

def load_model_for_question():
    if config.USE_OPENAI_FOR_QUESTION:
        print("  [INFO] 使用 OpenAI API 进行后半段问卷生成")
        return OpenAIModelWrapper(model_name=config.OPENAI_MODEL_NAME)
    else:
        print("  [INFO] 使用本地模型 进行后半段问卷生成")
        return LocalModelWrapper(model_path=config.MODEL_PATH_QUESTION, n_ctx=4096)

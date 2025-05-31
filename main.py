from fastapi import FastAPI, Request
from llama_cpp import Llama
import os
from pathlib import Path

app = FastAPI()

# 모델 로드 (최초 1회, 메모리에 유지)
model_path = "/home/ubuntu/bitnet_project/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# 모델 로드
llm = Llama(
    model_path=model_path,  # 반드시 문자열로 변환
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    # BitNet은 'system + user + assistant' 형식을 사용하는 chat_template가 있음
    full_prompt = f"System: You are a helpful assistant\nUser: {prompt}<|eot_id|>\nAssistant:"

    response = llm(full_prompt, max_tokens=128, stop=["<|eot_id|>"])
    return {"response": response["choices"][0]["text"].strip()}

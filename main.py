from fastapi import FastAPI, Request
import subprocess

app = FastAPI()

BITNET_EXEC = "/home/ubuntu/bitnet_project/BitNet/build/bin/llama-cli"
MODEL_PATH = "/home/ubuntu/bitnet_project/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# 시스템 역할 정의
SYSTEM_PROMPT = (
    "You are a helpful and concise assistant. "
    "Answer clearly without unnecessary detail. "
    "Avoid giving programming code unless explicitly asked.\n\n"
)

@app.get("/")
def read_root():
    return {"message": "Fast Server is running!"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_prompt = data.get("prompt", "")

    # 시스템 프롬프트 삽입
    full_prompt = f"{SYSTEM_PROMPT}User: {user_prompt}\nAssistant:"


    # 최대 토큰 제한
    cmd = [BITNET_EXEC, "-m", MODEL_PATH, "-p", full_prompt, "-n", "128"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        response_text = result.stdout.strip()

        # 반복 응답 제거
        if "The End" in response_text:
            response_text = response_text.split("The End")[0].strip()

    except subprocess.TimeoutExpired:
        return {"error": "Inference timed out"}
    except Exception as e:
        return {"error": f"{e}: {result.stderr if 'result' in locals() else ''}"}

    return {"response": response_text}

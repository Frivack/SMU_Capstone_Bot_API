from fastapi import FastAPI, Request
import subprocess

app = FastAPI()

BITNET_EXEC = "../BitNet/main"  # BitNet 실행파일 경로
MODEL_PATH = "../BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    cmd = [BITNET_EXEC, "-m", MODEL_PATH, "-p", prompt]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        response_text = result.stdout.strip()
    except Exception as e:
        return {"error": str(e)}

    return {"response": response_text}

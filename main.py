from fastapi import FastAPI, Request
import subprocess

app = FastAPI()

BITNET_EXEC = "/home/ubuntu/bitnet_project/BitNet/build/bin/llama-cli"
MODEL_PATH = "/home/ubuntu/bitnet_project/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

@app.get("/")
def read_root():
    return {"message": "Fast Server is running!"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    # 최대 토큰 제한 추가 (-n 128)
    cmd = [BITNET_EXEC, "-m", MODEL_PATH, "-p", prompt, "-n", "128"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        response_text = result.stdout.strip()

        # "The End" 반복 제거
        if "The End" in response_text:
            response_text = response_text.split("The End")[0].strip()

    except subprocess.TimeoutExpired:
        return {"error": "Inference timed out"}
    except Exception as e:
        return {"error": f"{e}: {result.stderr if 'result' in locals() else ''}"}

    return {"response": response_text}

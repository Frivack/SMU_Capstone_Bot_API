from fastapi import FastAPI, Request
import subprocess
import re

app = FastAPI()

BITNET_EXEC = "/home/ubuntu/bitnet_project/BitNet/build/bin/llama-cli"
MODEL_PATH = "/home/ubuntu/bitnet_project/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# 시스템 역할 정의
SYSTEM_PROMPT = (
    "You are a concise and knowledgeable assistant. "
    "Only respond with the answer to the user's input. "
    "Do not add emotional greetings, do not ask questions back, and do not translate or rephrase the user's input. "
    "Never say things like 'How can I help you?', 'Let me explain', or 'Is there something specific...'. "
    "Avoid emojis, filler phrases, or any form of repetition.\n"
    "Answer in natural, simple Korean unless English is required.\n"
)

@app.get("/")
def read_root():
    return {"message": "Fast Server is running!"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_prompt = data.get("prompt", "")

    # 시스템 프롬프트 삽입
    full_prompt = (
            SYSTEM_PROMPT +
            "User: 게임용 PC를 맞추고 싶은데, CPU는 어떤 게 좋아?\n"
            "Assistant: 게임용이라면 Intel i5-13600K나 AMD Ryzen 5 7600X 같은 중급 게이밍 CPU가 좋습니다.\n"
            "User: 3080에 어울리는 파워는?\n"
            "Assistant: RTX 3080은 최소 750W 이상의 정격 파워가 필요합니다. 브랜드는 시소닉이나 커세어가 안정적입니다.\n"
            "User: "
            f"{user_prompt}\nAssistant:"
    )

    # 최대 토큰 제한
    cmd = [BITNET_EXEC, "-m", MODEL_PATH, "-p", full_prompt, "-n", "80"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding='utf-8',
            errors='ignore',
            timeout=60
        )
        response_text = clean_response(result.stdout)

    except subprocess.TimeoutExpired:
        return {"error": "Inference timed out"}
    except Exception as e:
        return {
            "error": str(e),
            "stderr": result.stderr if 'result' in locals() else "no result"
        }
    return {"response": response_text}


def clean_response(text: str) -> str:
    # 마지막 Assistant 응답 추출
    matches = re.findall(r"Assistant:\s*(.*?)(?=\nUser:|\Z)", text, re.DOTALL)
    response = matches[-1].strip() if matches else text.strip()

    # [end of text] 제거
    response = response.replace("[end of text]", "").strip()

    # 이상한 문장 제거: 특수문자 혼합 단어
    response = re.sub(r'\b\w{15,}\b', '', response)


    # 이모지 제거 (간단 정규식)
    response = re.sub(r'[^\w\s.,?!\'\"()\-:+]', '', response)

    # 중복 라인 제거
    lines = response.splitlines()
    seen = set()
    response = "\n".join(line for line in lines if not (line in seen or seen.add(line)))

    return response.strip()
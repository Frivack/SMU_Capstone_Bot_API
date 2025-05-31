from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI, Request

app = FastAPI()

model_id = "microsoft/BitNet-b1.58-2B-4T"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=True)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    input_text = data.get("prompt", "")
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

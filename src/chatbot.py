from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "model/mistral-7b-instruct-v0.2-bnb-4bit"
lora_model = "model/Bajau-Mistral-7b-4bit"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base, lora_model)
model.eval()

def generate_response(messages):
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            do_sample=False
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

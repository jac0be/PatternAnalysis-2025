# from hugging face page on t5-base
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"
PROMPT = "Output exactly: 'ready ready ready'"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda"
model.to(device)

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
output = model.generate(**inputs)

print(tokenizer.decode(output[0], skip_special_tokens=True))

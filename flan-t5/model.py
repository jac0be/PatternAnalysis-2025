# from hugging face page on t5-base
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset import BioSummDataset

MODEL_NAME = "google/flan-t5-base"
PROMPT = "Output the following word 3 times: 'ready'"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda"
model.to(device)

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
output = model.generate(**inputs)

print(tokenizer.decode(output[0], skip_special_tokens=True))

# Try to generate summary of first report

ds = BioSummDataset(split="train")
rad_report, layman_sum = ds[0]

PROMPT = (
    "Summarize the following radiology report for a non-expert in 1-3 sentences. "
    "Be clear and avoid medical jargon.\n\nReport:\n"
    f"{rad_report}\n\nLayperson summary:"
)

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
output = model.generate(**inputs)
pred = tokenizer.decode(output[0], skip_special_tokens=True)

print("ORIGINAL REPORT \n", rad_report)
print("GENERATED \n", pred)
print("TRUE \n", layman_sum)

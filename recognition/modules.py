# modules.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

# returns fast tokenizer for seq2seq models
def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

# builds and returns the base-flan-t5 with attached LoRA adapters.
def build_flan_t5_with_lora(model_name="google/flan-t5-base", r=8, alpha=16, dropout=0.05):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        # NOTE: these match t5's projection layers
        target_modules=["q", "k", "v", "o"],
        bias="none",
    )

    return get_peft_model(model, cfg) # we convert the model to cuda outside this function, to prevent device mismatch issues
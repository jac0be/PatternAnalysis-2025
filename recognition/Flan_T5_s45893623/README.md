# FLAN-T5 + LoRA for Layperson Radiology Summarisation

This project fine-tunes **FLAN-T5-base** with **LoRA** adapters on the **BioLaySumm 2025 LaymanRRG** dataset.  
Given a radiology report, the model generates a 1–3 sentence summary in plain English that replaces medical jargon with basic concepts.

---

## 1. Problem Description

Radiology reports are written for clinicians and are difficult for patients to understand.  
The goal of this project is to:

- take an **expert radiology report** as input,  
- produce a **short layperson-friendly summary**,  
- evaluate performance using **ROUGE**,  
- and analyse where the model succeeds/fails.

We use a pretrained FLAN-T5-base model with LoRA adapters to efficiently fine-tune on an RTX 5070 TI (16GB)

## 2. Background

FLAN-T5 is a variant of Google's T5 that has been instruction-tuned on a large number of diverse tasks. Whilst the original T5 was already a strong encoder-decoder transformerm, FLAN-T5 is trained specifically to follow natural language instructions, making it more suitable at tasks phrased like "Rewrite this", "Summarise", or "Explain this".

### 2.1 Encoder/Decoder Architecture

FLAN-T5 uses the classic seq2seq structure:
- The encoder reads the input radiology report (plus our instruction prompt) and converts it into hidden representations.
- The decoder takes those representations and generates the summary token by token, handling both:
  - the encoder's information (what the report says)
  - previous generated tokens (what has already been written by the model)

FLAN-T5 is also relatively accessible compared to more complex LLMs, which require more VRAM to fine-tune.

### 2.2: LoRA (Low-Rank Adaptation)

FLAN-T5-base has ~250M parameters. Fully fine tuning all of them on a single consumer GPU is both slow and unneccesary.

**LoRA** adds a tiny number of trainable matrices inside the model's attention layers. Only these low-rank matrices are updated during training, which was around **1.7M parameters** in our configuration.

Effectively, **FLAN-T5** provides general reasoning ability and **LoRA** teaches it the domain-specific phrasing of radiology summaries.

## 



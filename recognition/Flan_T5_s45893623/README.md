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


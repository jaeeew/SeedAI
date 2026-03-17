#import

import random
import os
import torch
import shutil
from pathlib import Path

from datasets import load_dataset, Dataset
from itertools import islice
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


#setting

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B" 
OUT_DIR  = "./model"          

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

# Quantization
SCHEME = "W4A16"
TARGETS = ["Linear"]


#model loads

print("[INFO] 모델 로드 중…")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,  
    use_fast=True  
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()
torch.set_grad_enabled(False)

# ✅ 여기서 IGNORE 자동생성
IGNORE = [
    "lm_head", 
    "model.embed_tokens",
    "model.layers.0.mlp.down_proj",
    "model.layers.1.mlp.down_proj",
    "model.layers.28.mlp.down_proj",
    "model.layers.29.mlp.down_proj"
    ]


print("[INFO] IGNORE count:", len(IGNORE))
print("[INFO] sample IGNORE:", IGNORE[:10])

print("[INFO] 모델/토크나이저 로드 완료")

#dataset load&preprocess

print("[INFO] 캘리브레이션 데이터 로드 중...")

CANDIDATES = 7000  # 2000~8000 추천

raw = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=False)
raw = raw.shuffle(seed=42).select(range(CANDIDATES))

pairs = []
for ex in raw:
    text = tokenizer.apply_chat_template(
        ex["conversations"],
        add_generation_prompt=True,
        tokenize=False
    )
    tok_len = len(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH
        ).input_ids
    )
    pairs.append((tok_len, text))

pairs.sort(key=lambda x: x[0], reverse=True)
selected = pairs[:NUM_CALIBRATION_SAMPLES]

ds = Dataset.from_list([{"text": t} for _, t in selected])

print("[INFO] calib size:", len(ds))
print("[INFO] avg tok_len:", sum(l for l, _ in selected)/len(selected))

print("[INFO] 데이터 전처리 완료")

#Quantization

print(f"[INFO] GPTQ 시작 (scheme={SCHEME}, samples={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})...")

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        block_size = 64,
        targets=TARGETS,
        ignore=IGNORE,
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] GPTQ 완료")

#model save

os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

#submission

zip_name = "baseline_submit_UK"
print(f"[INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[INFO] 생성 완료: {zip_name}.zip")
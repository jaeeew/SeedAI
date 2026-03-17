import os
import shutil
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# ---------------------------
# Setting (Evaluation-friendly)
# ---------------------------
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "./model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# Calibration (speed/quality trade-off)
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 384

# Quantization
SCHEME = "W4A16"               # 4-bit weight, 16-bit activation
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]

# ---------------------------
# Runtime optimize
# ---------------------------
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Model Loads (lighter & faster)
# ---------------------------
print("[INFO] 모델 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=True,
)

# 일부 모델/템플릿에서 필요할 수 있어 안전하게 세팅
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,   # ✅ CPU RAM 절약
).to(device)

model.eval()

print("[INFO] 모델/토크나이저 로드 완료")

# ---------------------------
# Dataset Loads & Preprocess (memory-light)
# ---------------------------
print("[INFO] 캘리브레이션 데이터 로드 중...")

ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)

def preprocess(batch):
    texts = []
    for conv in batch["conversations"]:
        texts.append(
            tokenizer.apply_chat_template(
                conv,
                add_generation_prompt=True,
                tokenize=False
            )
        )
    return {"text": texts}

# ✅ batched + remove_columns로 전처리 속도/메모리 경량화
ds = ds.map(
    preprocess,
    batched=True,
    remove_columns=ds.column_names,
    desc="apply_chat_template",
)

print("[INFO] 데이터 전처리 완료")

# ---------------------------
# GPTQ Quantization
# ---------------------------
print(
    f"[INFO] GPTQ 시작 (scheme={SCHEME}, samples={NUM_CALIBRATION_SAMPLES}, "
    f"max_len={MAX_SEQUENCE_LENGTH})..."
)

recipe = [
    GPTQModifier(
        scheme=SCHEME,
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

# ---------------------------
# Model Save (submission-ready)
# ---------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# ✅ 제출 용량 감소 + 로딩 안정성
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

# ---------------------------
# Submission zip (structure must be: submit.zip/model/*)
# ---------------------------
zip_name = "submit"
print(f"[INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,   # zip 최상위에 model/ 이 오도록 유지
)

print(f"[INFO] 생성 완료: {zip_name}.zip")

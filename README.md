## 🚀 SeedAI: LG EXAONE 4.0-1.2B Quantization Project
본 프로젝트는 LG Aimers Phase 2 해커톤의 일환으로, EXAONE-4.0-1.2B 모델을 활용하여 성능 손실을 최소화하면서 추론 효율성을 극대화하는 경량화 솔루션을 개발하는 것을 목표로 합니다.

최근 On-device 및 제한된 환경에서의 AI 서비스 수요 증가에 따라 메모리 사용량, latency, 운영 비용을 줄이면서 성능을 유지하는 경량화 전략이 중요해졌습니다.

본 프로젝트에서는 단순 파라미터 감소가 아닌
GPTQ 기반 정밀 양자화 + 선택적 Layer 보호 전략을 적용하여 실제 vLLM 추론 환경에서의 효율 최적화를 목표로 설계했습니다.


## 👥 팀원 정보
팀명: SeedAI

구성원: 5명

## 🛠️ 개발 환경 (Environment)
1. Cloud Infrastructure
Platform: RunPod
GPU: NVIDIA L4 (24GB VRAM) - 대회 평가 환경과 동일하게 세팅
OS: Ubuntu 22.04 LTS (Docker Container)

2. Local Setup
Package Manager: Miniconda (Python 3.10)
IDE: VS Code (Remote - SSH 연결 필수)

## 🎯 과제 제약 사항 (Constraints)
대회 규정에 따라 아래 수치를 엄격히 준수합니다.

추론 시간: 전체 추론 시간이 20분 이내여야 함.

파일 용량: 제출 파일(submit.zip) 압축 해제 후 전체 크기가 32GB 이하일 것.

엔진: vLLM을 활용한 추론 최적화 (vLLM 라이브러리 자체 수정 금지).

## 방법론(Method)
1️⃣ Quantization Strategy

SCHEME = "W4A16"

TARGETS = ["Linear"]

- GPTQ 기반 4bit weight quantization
- 연산량이 큰 Linear layer 중심 적용

👉 속도 개선 + 메모리 절감 극대화

2️⃣ Selective Layer Protection (핵심 전략)

IGNORE = [

    "lm_head", 
    "model.embed_tokens",
    "model.layers.0.mlp.down_proj",
    "model.layers.1.mlp.down_proj",
    "model.layers.28.mlp.down_proj",
    "model.layers.29.mlp.down_proj"
]
✔️ 보호 대상
- 입력 임베딩
- 출력 레이어
- early / late layer 일부

👉 초기 표현 유지 + 출력 품질 유지

➡️ 정확도 저하 최소화 핵심

3️⃣ Calibration Data Optimization

CANDIDATES = 7000

NUM_CALIBRATION_SAMPLES = 512

MAX_SEQUENCE_LENGTH = 1024

✔️ 전략
- 데이터 7000개 샘플링
- 토큰 길이 기준 정렬
- 상위 512개 선택

pairs.sort(key=lambda x: x[0], reverse=True)

selected = pairs[:NUM_CALIBRATION_SAMPLES]

👉 긴 sequence 기반 calibration으로 attention weight 정밀도 확보

4️⃣ GPTQ Configuration

GPTQModifier(

    scheme=SCHEME,
    block_size=64,
    targets=TARGETS,
    ignore=IGNORE,
)

✔️ block_size = 64
성능 vs 속도 균형 최적화

5️⃣ One-shot Quantization

oneshot(

    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

👉 추가 학습 없이 빠른 경량화 수행

📊 Pipeline

MANTA Dataset

   ↓
   
Token Length 기반 정렬

   ↓
   
Top-K Calibration Sampling

   ↓

Selective Layer 보호

   ↓
   
GPTQ Quantization (W4A16)

   ↓
   
Compressed Model 저장 (HF format)

## 🤝 협업 및 보안 가이드 (Team Rules)
1. Git & GitHub 관리 (보안 필수)
대용량 파일 업로드 금지: *.safetensors, *.bin 등 모델 가중치 파일은 절대 push 하지 않습니다.
.gitignore 설정을 통해 로컬/서버의 대용량 파일을 보호합니다.
의존성 관리: 새로운 라이브러리 설치 시 반드시 pip freeze > requirements.txt를 갱신합니다.

2. RunPod 공용 계정 운영
인스턴스 종료: 작업이 끝나면 반드시 'Stop' 또는 **'Terminate'**를 확인하여 크레딧 낭비를 방지합니다.
데이터 보존: 중요한 작업물은 /workspace 또는 연결된 네트워크 볼륨에 저장하여 인스턴스 종료 시에도 유지되도록 합니다.

3. Hugging Face Access
모든 팀원은 개인 **Access Token (Read)**을 발급받아 huggingface-cli login을 마친 뒤 모델을 로드합니다.
모델 라이선스(exaone)를 확인하고 동의 절차를 완료해야 합니다.

## 📂 프로젝트 구조 (Proposed Structure)
Plaintext
SeedAI/
├── base_model/         # 원본 EXAONE 모델 가중치 (Git 제외)
├── quantized_model/    # 경량화 결과물이 저장될 폴더 (Git 제외)
├── scripts/            # 양자화 및 테스트 스크립트
│   ├── quantize_gptq.py
│   └── evaluate.ipynb
├── .gitignore          # 대용량 파일 및 보안 토큰 필터링
├── requirements.txt    # 환경 재현을 위한 라이브러리 목록
└── README.md           # 현재 파일

## 🚀 SeedAI: LG EXAONE 4.0-1.2B Quantization Project
본 프로젝트는 LG Aimers Phase 2 해커톤의 일환으로, EXAONE-4.0-1.2B 모델을 활용하여 성능 손실을 최소화하면서 추론 효율성을 극대화하는 경량화 솔루션을 개발하는 것을 목표로 합니다.

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

## 🚀 시작하기 (Quick Start)
Bash
###  1. 저장소 복제
git clone https://github.com/DeiLee0913/SeedAI.git

### 2. 가상 환경 활성화
conda activate aimers

### 3. 필수 라이브러리 설치
pip install -r requirements.txt

### 4. 허깅페이스 로그인
huggingface-cli login

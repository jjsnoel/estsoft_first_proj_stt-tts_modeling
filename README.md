# VoiceProject — 실시간 다국어 통역 시스템

외국인 손님과 한국인 직원 사이의 언어 장벽을 실시간으로 해소하는 음성 파이프라인입니다.  
1인 자영업 카페·식당 환경에 최적화되어 있으며, 스마트폰·태블릿만으로 구동 가능합니다.

---

## 주요 기능

- **손님 모드**: 외국어 음성 → STT(자동 감지) → 한국어 번역 → 주문 슬롯 추출 → 화면 자막 출력
- **직원 모드**: 한국어 음성 → STT → 손님 언어로 번역 → TTS(보이스클론) → 음성 출력
- **보이스클론**: 사장님 목소리를 학습해 알바생이 응대해도 동일한 브랜드 음성으로 송출
- **주문 슬롯 파싱**: 메뉴·옵션·수량·포장 여부 등 자동 구조화 (`keyword_mapper.py`)
- **불가 요청 거절**: OpenAI 번역 프롬프트에 정중한 거절 톤 내장

지원 언어: 영어, 중국어(간체), 일본어, 프랑스어, 스페인어

---

## 파이프라인 구조

```
[손님 모드]
마이크 입력 (외국어)
    └─▶ faster-whisper STT (언어 자동 감지)         ~1.0s (GPU)
            └─▶ GPT-4.1-mini 번역 → 한국어          ~2.6s
                    └─▶ keyword_mapper 주문 슬롯 파싱
                            └─▶ Gradio 화면 자막 출력

[직원 모드]
마이크 입력 (한국어)
    └─▶ faster-whisper STT (ko 고정)                ~1.0s (GPU)
            └─▶ GPT-4.1-mini 번역 → 손님 언어        ~2.6s
                    └─▶ edge-tts 음성 생성 (페르소나)  ~0.5s
                            └─▶ OpenVoice V2 보이스클론 ~1.7s (GPU)
                                    └─▶ 스피커 출력
```

| 단계 | 모델 | GPU 처리시간 | CPU 처리시간 |
|------|------|-------------|-------------|
| STT 모델 로드 | faster-whisper-medium | 2.0s | 26.8s |
| STT 추론 | faster-whisper-medium | 1.0s | 15.6s |
| 번역 | GPT-4.1-mini | 2.6s | 2.6s (동일) |
| TTS 생성 | edge-tts | 0.5s | 0.7s (동일) |
| 보이스클론 | OpenVoice V2 | 1.7s | 3.8s |
| **전체** | | **8.8s** | **49.3s** |

---

## 폴더 구조

```
VoiceProject/
├── app.py                  ← Gradio 통합 앱 (메인 실행 파일)
├── download_models.py      ← HuggingFace 모델 자동 다운로드
├── requirements.txt        ← STT + 번역 의존성 (Python 3.11)
├── requirements_tts.txt    ← TTS + OpenVoice 의존성 (Python 3.10)
├── setup.bat               ← .venv 자동 세팅
├── create_folders.bat      ← 하위 폴더 일괄 생성
├── run_pipeline.bat        ← 전체 파이프라인 실행
├── .env                    ← API 키 설정 (OPENAI_API_KEY)
│
├── scripts/
│   ├── stt.py              ← 1단계: 음성 → 텍스트
│   ├── translate.py        ← 2단계: 텍스트 번역
│   ├── tts.py              ← 3단계: 텍스트 → 음성
│   └── keyword_mapper.py   ← 주문 슬롯 파싱 (카페 도메인)
│
├── models/
│   ├── whisper/            ← faster-whisper-medium
│   ├── m2m100/             ← 오프라인 번역 fallback (facebook/m2m100_418M)
│   └── openvoice/          ← 보이스클론 (OpenVoice V2)
│
├── reference_voices/       ← 보이스클론용 참조 음성 (.wav / .pth)
├── train_data/             ← STT 테스트용 오디오 (en, zh)
├── output/
│   ├── stt_result/         ← STT 결과 JSON
│   ├── translated/         ← 번역 결과 JSON
│   └── tts_audio/          ← 생성된 음성 파일
└── logs/                   ← 실행 로그 (일별 rotation)
```

---

## 환경 구성

Python 버전 충돌로 가상환경 2개를 분리 운용합니다.

| 환경 | Python | 용도 |
|------|--------|------|
| `.venv` | 3.11 | STT + 번역 + Gradio 앱 전체 |
| `.venv310` | 3.10 | TTS (edge-tts + OpenVoice V2) |

### 최초 세팅 순서

**1. 폴더 생성**
```powershell
.\create_folders.bat
```

**2. .venv 세팅 (STT + 번역 + 앱)**
```powershell
.\setup.bat
```

**3. .venv310 세팅 (TTS)**
```powershell
python -m venv .venv310
.venv310\Scripts\activate
pip install -r requirements_tts.txt
```

**4. OpenVoice 설치**
```powershell
pip install -e OpenVoice/ --no-deps
```

**5. HuggingFace 모델 다운로드**
```powershell
.venv\Scripts\activate
python download_models.py --all
```

**6. Whisper 모델 다운로드**
```powershell
huggingface-cli download Systran/faster-whisper-medium --local-dir models\whisper
```

**7. OpenVoice 체크포인트 복사**
```
models\openvoice\checkpoints_v2\ 에 OpenVoice V2 체크포인트 배치
reference_voices\ 에 참조 음성 파일 배치 (.wav 또는 .pth)
```

**8. 환경변수 설정**
```
.env 파일 생성 후 아래 내용 입력:
OPENAI_API_KEY=sk-...
```

---

## 실행 방법

### Gradio 앱 실행 (권장)
```powershell
.venv\Scripts\activate
python app.py
```
브라우저에서 `http://127.0.0.1:7860` 접속

### 전체 파이프라인 CLI 실행
```powershell
# 보이스클론 없이
.\run_pipeline.bat input\test.wav

# 보이스클론 적용
.\run_pipeline.bat input\test.wav reference_voices\woman_voice_combined.wav
```

### 단계별 개별 실행
```powershell
# 1단계 STT
.venv\Scripts\activate
python scripts\stt.py --input input\test.wav

# 2단계 번역
python scripts\translate.py
python scripts\translate.py --tgt en   # 목표 언어 강제 지정

# 3단계 TTS
deactivate
.venv310\Scripts\activate
python scripts\tts.py
python scripts\tts.py --voice_ref reference_voices\woman_voice_combined.wav
python scripts\tts.py --no_clone      # 보이스클론 없이 기본 음성만
```

---

## 주요 모듈 설명

### `app.py` — Gradio 통합 앱
손님 탭과 직원 탭으로 구성된 메인 앱. warmup 시 모든 모델을 사전 로드하며, 이후 요청은 캐싱된 모델을 재사용합니다.

```python
# 핵심 함수 3개
run_stt(audio_path, language)      # STT 실행 + 주문 슬롯 파싱
run_translate(text, src, tgt)      # OpenAI 우선, m2m100 fallback
run_tts(text, language, voice_ref) # edge-tts + OpenVoice 보이스클론
```

### `scripts/keyword_mapper.py` — 주문 슬롯 파서
카페 도메인 특화 정규식 기반 파서. 영어·한국어·중국어 패턴을 동시 처리하며, 보조 질문(Yes/No 응답 추론), 결제 문맥 오인식 보정, 직원 응답 필터링 기능을 포함합니다.

```python
extract_order_keywords(text)   # 키워드 매칭 + 중복 제거
build_order_slots(keywords)    # 구조화된 주문 슬롯 생성
build_order_slots_ko(slots)    # 한국어 레이블로 변환
strip_staff_response_text(text) # 직원 발화 필터링
```

### 번역 우선순위
1. **OpenAI GPT-4.1-mini**: 카페 맥락 프롬프트 적용, 자연스러운 번역
2. **m2m100_418M (fallback)**: 오프라인 환경 또는 API 키 없을 때

---

## 주요 모델 출처

| 모델 | 용도 | 출처 |
|------|------|------|
| faster-whisper-medium | STT | [HuggingFace - Systran](https://huggingface.co/Systran/faster-whisper-medium) |
| facebook/m2m100_418M | 다국어 번역 fallback | [HuggingFace - Facebook](https://huggingface.co/facebook/m2m100_418M) |
| OpenVoice V2 | 보이스클론 TTS | [MyShell - OpenVoice](https://github.com/myshell-ai/OpenVoice) |
| edge-tts | 기본 TTS | Microsoft Neural TTS |
| GPT-4.1-mini | 번역 (메인) | OpenAI API |

---

## 참고 사항

- FFmpeg가 필요합니다. `C:\ffmpeg\bin\` 경로에 설치하거나 `app.py` 내 `FFMPEG_BIN` 경로를 수정하세요.
- GPU(CUDA) 환경에서 전체 처리시간 약 8.8초, CPU 환경에서 약 49.3초입니다.
- OpenAI API 키가 없으면 번역은 m2m100 오프라인 모델로 자동 전환됩니다.
- 보이스클론 참조 음성은 `.wav` 원본 또는 사전 추출된 `.pth` 임베딩 모두 사용 가능합니다.

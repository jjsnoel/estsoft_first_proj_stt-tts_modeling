@echo off
REM VoiceProject 하위 폴더 전체 생성 스크립트
REM setup.bat 이후 실행하거나 setup.bat에 통합 가능

mkdir models\whisper        2>nul
mkdir models\whisper_tuned  2>nul
mkdir models\m2m100         2>nul
mkdir models\openvoice      2>nul
mkdir input                 2>nul
mkdir output\stt_result     2>nul
mkdir output\translated     2>nul
mkdir output\tts_audio      2>nul
mkdir reference_voices      2>nul
mkdir logs                  2>nul

echo 폴더 구조 생성 완료!
echo.
echo 다음 단계:
echo  1. models\ 폴더에 각 모델 파일 복사
echo  2. input\ 폴더에 테스트 오디오 파일 복사
echo  3. setup.bat 실행하여 가상환경 및 패키지 설치

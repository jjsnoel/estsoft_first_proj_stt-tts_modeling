@echo off
setlocal enabledelayedexpansion
echo ============================================
echo   VoiceProject Pipeline (Improved)
echo ============================================
REM =========================
REM Input Check
REM =========================
set INPUT_FILE=%1
set VOICE_REF=%2
if "%INPUT_FILE%"=="" (
    echo [ERROR] Usage: run_pipeline.bat input.wav [voice.wav]
    exit /b 1
)
if not exist "%INPUT_FILE%" (
    echo [ERROR] File not found: %INPUT_FILE%
    exit /b 1
)
echo Input: %INPUT_FILE%
echo.
REM =========================
REM [1] STT + Translate
REM =========================
echo [1/3] STT  (.venv)
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found
    exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\stt.py --input "%INPUT_FILE%"
if errorlevel 1 (
    echo [ERROR] STT failed
    call deactivate
    exit /b 1
)

echo [2/3] Translate

python scripts\translate.py
if errorlevel 1 (
    echo [ERROR] Translate failed
    call deactivate
    exit /b 1
)
:: 1. 파이썬 코드를 변수에 할당 (따옴표로 감싸서 특수문자 '='와 '()' 충돌 방지)
set "PY_CODE=import json, glob, os; f=sorted(glob.glob('output/translated/*.json'), key=os.path.getmtime, reverse=True); print(json.load(open(f[0], encoding='utf-8'))['tgt_lang']) if f else print('none')"

:: 2. FOR /F 문에서 변수를 사용하여 안전하게 실행
for /f "usebackq delims=" %%i in (`python -c "%PY_CODE%"`) do (
    set "tgt_LANG=%%i"
)

echo Detected source language: %tgt_LANG%

call deactivate

REM =========================
REM [3] TTS (Simplified)
REM =========================
if not "%tgt_LANG%"=="ko" (
    echo.
    echo [3/3] TTS (.venv310)

    if not exist ".venv310\Scripts\activate.bat" (
        echo [ERROR] .venv310 not found
        exit /b 1
    )

    call .venv310\Scripts\activate.bat

    :: 한국어일 때만 TTS 실행 (voice clone 제거)
    python scripts\tts.py 

    if errorlevel 1 (
        echo [ERROR] TTS failed
        call deactivate
        exit /b 1
    )

    call deactivate
) else (
    echo.
    echo [3/3] Guest mode: Skipping TTS.
    echo Check subtitles in: output\translated\
)
REM =========================
REM Done
REM =========================
echo.
echo ============================================
echo   Pipeline complete!
echo   STT    : output\stt_result\
echo   Trans  : output\translated\
echo   TTS    : output\tts_audio\
echo ============================================
pause
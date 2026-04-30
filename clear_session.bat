@echo off
echo Clearing session data...
if exist output\stt_result del /q output\stt_result\*
if exist output\translated del /q output\translated\*
if exist output\tts_audio del /q output\tts_audio\*
if exist logs\last_foreign_lang.txt del /q logs\last_foreign_lang.txt
echo Session cleared! Ready for demo.
pause

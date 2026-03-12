@echo off
chcp 65001 >nul
echo 批量声纹注册工具
echo 将处理 speaker_embeddings 文件夹下的所有 PCM 文件，并为每个文件生成对应的 npz 文件
echo.

REM 使用项目自带的 Python 解释器
set PYTHON_EXE=.\runtime\python.exe

REM 检查 Python 是否可用
%PYTHON_EXE% --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到 %PYTHON_EXE%，请确保路径正确
    pause
    exit /b
)

REM 运行批量转换脚本（注意：原脚本不接受 --output 参数）
%PYTHON_EXE% batch_pcm_to_npz.py --folder "speaker_embeddings"

echo.
echo 批量处理完成！生成的 npz 文件与对应的 pcm 文件在同一文件夹下。
pause
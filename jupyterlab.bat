@echo off
call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
call conda activate cnn
call jupyter-lab
pause

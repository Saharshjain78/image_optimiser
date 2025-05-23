@echo off
echo Starting Neural Noise Segmentation App...
cd /d "c:\Users\Asus\Documents\Projects\image_optimisation"
powershell.exe -Command "& {pip install -r requirements.txt; streamlit run streamlit_app.py}"
pause

@echo off
cd /d "c:\Users\Asus\Documents\Projects\image_optimisation"
echo Running debug script... > debug_log.txt
python debug_app.py >> debug_log.txt 2>&1
echo Running application... >> debug_log.txt
python -m streamlit run streamlit_app.py >> debug_log.txt 2>&1

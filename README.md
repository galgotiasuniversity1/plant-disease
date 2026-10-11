# Leaf Disease Detector (Flask + HTML/CSS/JS)
Owner: Pramod Singh Gaira

## Run
    pip install -r requirements.txt
    cp .env.example .env     # add GROQ_API_KEY (optional: the page can also ask for it)
    python server.py         # open http://127.0.0.1:5000

## Structure
- server.py        Flask API: /api/check, /api/analyze, /api/report
- core.py          leaf check, lighting check, Groq call, heatmap, PDF
- static/          index.html, style.css, app.js

## Flow
Browse/drag-drop or camera -> local leaf + lighting checks (nothing sent to AI if they fail)
-> Analyze -> Plant/Condition/Confidence/Severity, tabs, Top Predictions, heatmap -> PDF.

Notes: the camera needs http://localhost or HTTPS. The heatmap is colour analysis, not true Grad-CAM
(Groq is API-only). Vision models tried: qwen/qwen3.8-27b, then qwen/qwen3.6-27b.

import subprocess
import sys
import os
import threading
import http.server
import webbrowser
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # goes up to project root
APP_FILE     = PROJECT_ROOT / "app" / "streamlit_app.py"
WEB_DIR      = PROJECT_ROOT / "app" / "nse_alpha_web"   # folder with index.html & login.html

def start_streamlit():
    """Start Streamlit on port 8501"""
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP_FILE),
         "--server.port=8501", "--server.headless=true"],
        cwd=str(PROJECT_ROOT)
    )

def start_web_server():
    """Serve index.html + login.html on port 8502"""
    os.chdir(WEB_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    server  = http.server.HTTPServer(("localhost", 8502), handler)
    server.serve_forever()

if __name__ == "__main__":
    print("🚀 Starting Streamlit dashboard on port 8501...")
    start_streamlit()

    print("🌐 Starting web server (landing page) on port 8502...")
    t = threading.Thread(target=start_web_server, daemon=True)
    t.start()

    time.sleep(2)
    print("✅ Opening browser...")
    webbrowser.open("http://localhost:8502")  # opens landing page

    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down.")
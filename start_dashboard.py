"""Launch Streamlit on the hosting platform's assigned port."""
import os
import sys


os.execv(sys.executable, [
    sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
    "--server.address", "0.0.0.0",
    "--server.port", os.getenv("PORT", "8000"),
    "--server.headless", "true",
])
#!/usr/bin/env python3
"""
run.py — convenience launcher
Usage:
  python run.py ui          # start Streamlit
  python run.py api         # start FastAPI
  python run.py both        # start both (in background)
  python run.py test        # run test suite
"""

import sys
import subprocess


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "ui"

    if mode == "ui":
        print("Starting Streamlit UI on http://localhost:8501")
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"])

    elif mode == "api":
        print("Starting FastAPI on http://localhost:8000")
        print("Docs: http://localhost:8000/docs")
        subprocess.run(["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

    elif mode == "both":
        import threading
        import time

        def run_api():
            subprocess.run(["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])

        def run_ui():
            time.sleep(2)
            subprocess.run(["streamlit", "run", "app/streamlit_app.py"])

        t1 = threading.Thread(target=run_api, daemon=True)
        t2 = threading.Thread(target=run_ui)
        t1.start()
        t2.start()
        t2.join()

    elif mode == "test":
        subprocess.run([sys.executable, "tests/test_pipeline.py"])

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python run.py [ui|api|both|test]")
        sys.exit(1)


if __name__ == "__main__":
    main()

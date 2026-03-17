import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
exec(open("app/streamlit_app.py").read())
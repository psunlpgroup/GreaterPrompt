import subprocess
import sys


def main():
    script_path = "Web/🏠Overview.py"
    sys.exit(subprocess.run(["streamlit", "run", script_path]).returncode)

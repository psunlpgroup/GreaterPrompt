import os
import subprocess
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "Web", "🏠Overview.py")
    sys.exit(subprocess.run(["streamlit", "run", script_path]).returncode)


if __name__ == "__main__":
    main()

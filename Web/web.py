import subprocess
import sys


def main():
    sys.exit(subprocess.run(["streamlit", "run", "🏠Overview.py"]).returncode)


if __name__ == "__main__":
    main()

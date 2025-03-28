# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README_pypi.md").read_text(encoding="utf-8")

setup(
    name="greaterprompt",
    version="1.1.1",
    description="A Gradient-based Prompt Optimizer for Text Generation",
    long_description_content_type="text/markdown",
    url="https://github.com/WenliangZhoushan/GreaterPrompt",
    author="Wenliang Zheng",
    author_email="wenliangz2004@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="greaterprompt, gradient-based, prompt, optimizer, text, generation",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["greaterprompt*"]),
    python_requires="==3.11",
    install_requires=[
        "altair", "accelerate", "autopep8==2.3.2", "black==25.1.0", "datasets==3.3.2", 
        "diskcache==5.6.3", "fschat", "graphviz==0.20.3", "guidance==0.0.64", 
        "httpx==0.28.1", "numpy", "pandas==2.2.3", "Pillow==11.1.0", "platformdirs==4.3.6", 
        "psutil", "protobuf==5.29.4", "pytest==8.3.5", "python-dotenv==1.0.1", 
        "python_Levenshtein==0.27.1", "Requests==2.32.3", "retrying==1.3.4", 
        "setuptools==75.8.0", "streamlit==1.43.0", "tenacity==9.0.0", "torch==2.3.1", 
        "tqdm==4.67.1", "transformers==4.48.3", "vllm==0.5.3.post1", "openai==0.27.8"
    ],
    entry_points={
        "console_scripts": [
            "greaterprompt=greaterprompt.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/WenliangZhoushan/GreaterPrompt/issues",
        "Source": "https://github.com/WenliangZhoushan/GreaterPrompt/",
    },
)

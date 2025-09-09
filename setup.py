from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="live-captions-translator",
    version="1.3.0",
    author="Live Captions Translator Team",
    author_email="",
    description="Real-time caption translation with intelligent sentence boundary detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/live-captions-translator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video :: Display",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "live-captions-translator=Caption_live_translater:run_gui",
        ],
    },
    keywords="live captions translation ocr real-time windows accessibility",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/live-captions-translator/issues",
        "Source": "https://github.com/yourusername/live-captions-translator",
    },
)

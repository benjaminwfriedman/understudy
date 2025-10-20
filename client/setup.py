from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="understudy-client",
    version="1.0.0",
    author="Understudy Team",
    author_email="team@understudy.ai",
    description="Python client for Understudy - LLM to SLM training platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/understudy/understudy-client",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langgraph>=0.0.20",
        "typing-extensions>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0",
            "ruff>=0.1.0",
        ]
    },
)
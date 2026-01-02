"""
Setup script for Curio Agent SDK.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="curio-agent-sdk",
    version="0.1.0",
    author="Curio Team",
    description="A flexible, model-agnostic agentic framework for building autonomous AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ujjalsharma100/curio-agent-sdk",
    package_dir={"curio_agent_sdk": "."},
    packages=["curio_agent_sdk", "curio_agent_sdk.config", "curio_agent_sdk.core", "curio_agent_sdk.llm", "curio_agent_sdk.llm.providers", "curio_agent_sdk.persistence"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "groq": ["groq>=0.4.0"],
        "ollama": ["ollama>=0.1.0"],
        "postgres": ["psycopg2-binary>=2.9.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "groq>=0.4.0",
            "ollama>=0.1.0",
            "psycopg2-binary>=2.9.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "curio-agent=curio_agent_sdk.cli:main",
        ],
    },
)

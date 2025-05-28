# FinGPT Repository Overview

## Introduction

FinGPT is an open-source project dedicated to advancing the use of Large Language Models (LLMs) in the financial domain. It aims to provide cost-effective, adaptable, and democratized access to financial data and LLM capabilities. The project emphasizes lightweight adaptation of existing open-source LLMs, fine-tuning them for specific financial tasks, and leveraging techniques like Reinforcement Learning from Human Feedback (RLHF) to enhance their performance and alignment with user preferences.

This document provides an overview of the repository's structure and the purpose of its main components.

## Top-Level Project Files and Directories

The root of the FinGPT repository contains several key files and directories:

*   **`README.md`**: The main entry point and comprehensive guide to the FinGPT project, outlining its goals, architecture, key applications, and resources.
*   **`.github/`**: Contains GitHub-specific files for project management and community interaction, including:
    *   `FUNDING.yml`: Information on how to financially support the project.
    *   `ISSUE_TEMPLATE/`: Standardized templates for reporting issues or requesting features.
*   **`.gitignore`**: Specifies files and directories that Git should ignore, such as compiled Python files (`*.pyc`) or local environment folders.
*   **`.gitpod.yml`**: Configuration file for Gitpod, enabling a ready-to-code development environment in the cloud, typically by automating dependency installation.
*   **`CODE_OF_CONDUCT.md`**: Outlines the expected standards of behavior for community members and contributors.
*   **`CONTRIBUTING.md`**: Provides guidelines for developers looking to contribute to the FinGPT project.
*   **Jupyter Notebooks (Root Level)**:
    *   **`FinGPT_Inference_Llama2_13B_falcon_7B_for_Beginners.ipynb`**: A tutorial demonstrating how to use pre-trained FinGPT models (Llama2-13B for sentiment analysis, Falcon-7B for multi-task NLP) for inference.
    *   **`FinGPT_Training_LoRA_with_ChatGLM2_6B_for_Beginners.ipynb`** (and `_v2-2.ipynb`): Tutorial notebooks guiding users on fine-tuning the ChatGLM2-6B model using LoRA for financial tasks.
*   **`LICENSE`**: Contains the MIT License under which the FinGPT project is distributed.
*   **`MANIFEST.in`**: Used by Python's `setuptools` to specify which files to include in source distributions of the FinGPT library.
*   **`figs/`**: A directory storing images, diagrams, and plots used in the `README.md` and other documentation.
*   **`fingpt/`**: The main Python package directory containing the core source code for various FinGPT modules and applications.
*   **`requirements.txt`**: Lists top-level Python dependencies. Specific sub-modules may have their own `requirements.txt`.
*   **`setup.py`**: The standard Python script for packaging and distributing the `FinGPT` library.

## The `fingpt/` Package: Core Modules

The `fingpt/` directory is the central package housing all core FinGPT functionalities.

*   **`fingpt/__init__.py`**: Makes `fingpt/` a Python package.
*   **`fingpt/readme.md`**: An internal table of contents for the sub-modules within the `fingpt` package.

### Key Submodules within `fingpt/`:

*   **`FinGPT_Benchmark/`**:
    *   **Purpose**: Provides a framework for benchmarking open-source LLMs on financial NLP tasks using instruction tuning.
    *   **Key Components**:
        *   `train_lora.py`: Script for LoRA-based fine-tuning of LLMs.
        *   `benchmarks/benchmarks.py`: Script for evaluating models on various financial benchmarks.
        *   `benchmarks/*.py`: Individual scripts for specific benchmark datasets (e.g., `fpb.py`, `fiqa.py`).
        *   `data/`: Scripts and notebooks for dataset preparation.
        *   `config*.json`: DeepSpeed configuration files for training.
        *   `utils.py`: Utility functions for benchmarking.
        *   `demo.ipynb`: Notebook for demonstrating inference with multi-task models.

*   **`FinGPT_FinancialReportAnalysis/`**:
    *   **Purpose**: Tools for analyzing financial reports (e.g., 10-K SEC filings) using LLMs.
    *   **Key Components**:
        *   `reportanalysis.ipynb`: A Jupyter Notebook that orchestrates data fetching (company info, stock data, SEC filings), LLM-based analysis (of income statements, balance sheets, cash flows), and generation of a PDF summary report using ReportLab.
        *   `utils/`: Helper scripts including:
            *   `earning_calls.py`: For fetching earnings call transcripts.
            *   `rag.py`: Implements a `Raptor` class, likely for advanced RAG techniques.

*   **`FinGPT_Forecaster/`**:
    *   **Purpose**: Implements the FinGPT-Forecaster for predicting stock price movements based on news and financial data.
    *   **Key Components**:
        *   `app.py`: A Gradio web application for interactive forecasting demos.
        *   `train_lora.py`: Script for LoRA fine-tuning of LLMs (Llama-2, ChatGLM2) for the forecasting task.
        *   Data handling scripts (`data.py`, `data_pipeline.py`, etc.) for fetching and processing data.
        *   `prompt.py`: Manages prompt templates for forecasting.
        *   `config.json`: DeepSpeed configuration for training.
        *   `FinGPT-Forecaster-Chinese/`: Sub-module for Chinese market forecasting.

*   **`FinGPT_MultiAgentsRAG/`**:
    *   **Purpose**: Explores the use of Multi-Agent Systems (MAS) and Retrieval-Augmented Generation (RAG) to enhance the factual accuracy of LLMs in finance.
    *   **Key Components**:
        *   `Evaluation_methods/`: Tools for evaluating LLM factuality (HaluEval, MMLU, TruthfulQA).
        *   `Fine_tune_model/`: Notebooks for fine-tuning agent models.
        *   `MultiAgents/` & `RAG/`: Notebooks for experimenting with multi-agent and RAG setups.

*   **`FinGPT_Others/`**:
    *   **Purpose**: A collection of other experimental FinGPT projects.
    *   **Key Components**: Includes sub-projects like `FinGPT_Low_Code_Development/`, `FinGPT_Robo_Advisor/`, and `FinGPT_Trading/`.

*   **`FinGPT_RAG/`**:
    *   **Purpose**: Focuses on enhancing financial sentiment analysis using a Retrieval Augmented Generation (RAG) framework.
    *   **Key Components**:
        *   `multisource_retrieval/news_scraper.py`: A script to scrape news from numerous financial websites to provide external context to the LLM.
        *   `scrapers/`: Specific scraping logic for different news sites.
        *   `instruct-FinGPT/train.py`: A wrapper script for launching supervised fine-tuning of the LLM used in the RAG system.

*   **`FinGPT_Sentiment_Analysis_v1/`**:
    *   **Purpose**: Early versions of sentiment analysis models where training labels were derived from market stock price changes ("labeled by the market").
    *   **Key Components**: Contains code and examples for FinGPT v1.0 (ChatGLM2 for Chinese market) and v1.1 (Llama-2-13B for US market).

*   **`FinGPT_Sentiment_Analysis_v3/`**:
    *   **Purpose**: More advanced sentiment analysis models fine-tuned with LoRA on academic and GPT-labeled datasets.
    *   **Key Components**:
        *   `README.md`: Provides usage examples and extensive benchmark results.
        *   `benchmark/`: Notebooks to reproduce benchmark scores.
        *   `data/making_data.ipynb`: Notebook for training data preparation.
        *   `training_8bit/`, `training_int4/`: Notebooks for training with 8-bit and 4-bit (QLoRA) quantization.
        *   `training_parallel/train_lora.py`: Script for parallel LoRA fine-tuning (e.g., with ChatGLM2).
        *   `training_parallel/config.json`: DeepSpeed configuration for parallel training.

This overview should help in navigating the FinGPT repository and understanding the role of its various components.

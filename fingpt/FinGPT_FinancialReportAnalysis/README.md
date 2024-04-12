# Financial Report Analysis Project

## Overview

This project provides tools for analyzing financial reports, specifically annual reports (10-K), using advanced language models such as GPT-4 or other locally deployed Large Language Models (LLM). It's designed to help users generate comprehensive analysis reports in PDF format, offering insights into a company's financial health and performance over the fiscal year.

## Features

- **PDF Report Generation**: Automatically generate detailed analysis reports in PDF format for annual financial statements.
- **GPT-4 and LLM Support**: Utilize the power of GPT-4 or any locally deployed LLM for deep and insightful analysis.
- **RAG Support**: The ability to utilize the power of RAG for question-answering and summarization tasks.
- **Customizable Analysis**: Users can modify the analysis scope by choosing different company symbols and models.
- **Easy to Use**: Designed with simplicity in mind, simply run all cells in the provided notebook to get your report.

## Requirements

Before starting, ensure you have the following installed:
- Python 3.11 or later
- Jupyter Notebook
- Necessary Python packages (pandas, matplotlib, openai, etc.)

Obtain the sec-api (which is used to grab the 10-K report) from https://sec-api.io/profile for free.

(Optional) Obtain the fmp api for target price (paid) from https://site.financialmodelingprep.com/developer/docs/dashboard.

## Getting Started

To begin analyzing financial reports:

0. **(optional) Prepare the local LLM**:
   If you want to run the analysis with the locally deployed models, please download Ollama and have it running: https://ollama.com/download.
   Also, download the model you want to use in the list of available models: https://ollama.com/library with command:
   ```bash
    ollama run <model_name>
    ```

1. **Open the Notebook**:
   Launch Jupyter Notebook and open the `reportanalysis.ipynb` file:
   ```
   jupyter notebook reportanalysis.ipynb
   ```
   All the necessary libraries and dependencies are already imported in the notebook.

2. **Configure the Notebook**:
   Modify the `company symbol` and `models` variables within the notebook to suit the analysis you wish to perform.

3. **Run the Analysis**:
   Execute all cells in the notebook to generate your financial report analysis in PDF format.

## Contributing

We welcome contributions and suggestions! Please open an issue or submit a pull request with your improvements.

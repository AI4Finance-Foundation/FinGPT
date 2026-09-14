# FinGPT Frequently Asked Questions (FAQ)

This document addresses common questions about FinGPT usage, capabilities, and limitations.

## General Usage Questions

### Can I use FinGPT to extract financial information from financial reports?

**Answer**: FinGPT is primarily designed for financial sentiment analysis and forecasting tasks. While it can process financial text, extracting structured financial information (metrics, values, company names) from reports is not its primary function. For information extraction tasks, you may need to:

1. Use FinGPT for sentiment analysis on the extracted information
2. Combine with specialized NER (Named Entity Recognition) models
3. Use the multi-task FinGPT models which include some entity recognition capabilities
4. Consider custom fine-tuning for your specific extraction needs

The multi-task models available on HuggingFace include financial named entity recognition as one of the tasks.

### Can I use FinGPT for forecasting other markets (e.g., China stock market)?

**Answer**: FinGPT-Forecaster was primarily trained on US market data (DOW 30). While the underlying architecture could be adapted for other markets, the current models are optimized for US market patterns. For China stock market forecasting:

1. You would need to collect appropriate Chinese market data
2. Fine-tune the model on Chinese market data
3. Consider using FinGPT v1.x models which were designed for Chinese markets
4. The data pipeline and model architecture would need adaptation for Chinese market characteristics

We welcome contributions for extending market coverage!

### What are the hardware requirements for training FinGPT?

**Answer**: Training requirements vary by model size and dataset:

**Minimum Requirements:**
- For inference: CPU + 8GB RAM (cloud API recommended)
- For fine-tuning small models: NVIDIA GPU with 12GB+ VRAM

**Recommended for Full Training:**
- GPU: NVIDIA RTX 3090, A100, or equivalent
- VRAM: 24GB+ for 13B models, 12GB+ for 7B models
- RAM: 32GB+ system RAM
- Storage: 50GB+ SSD

**Cloud Options:**
- Google Colab (free tier with GPU access)
- Kaggle Kernels (free GPU access)
- RunPod, Vast.ai, Lambda Labs (affordable GPU rentals)

**Training Time Estimates:**
- FinGPT v3.3 (13B) on RTX 3090: ~17 hours
- FinGPT v3.2 (7B) on A100: ~5.5 hours
- Smaller datasets can train in significantly less time

### Can I train FinGPT on Google Colab with A100?

**Answer**: Yes, you can train FinGPT on Google Colab with A100 GPU, but with some limitations:

1. **Time Limits**: Colab has time limits on free and paid tiers
2. **Model Size**: A100 can handle 13B models, but training time may be limited
3. **Storage**: Colab storage is temporary - you'll need to save models to Google Drive
4. **Memory**: A100 (40GB) can handle most training configurations

**Recommendations for Colab Training:**
- Use smaller model variants (7B instead of 13B)
- Enable gradient checkpointing to reduce memory usage
- Use 8-bit or 4-bit quantization
- Save checkpoints frequently to Google Drive
- Consider using Colab Pro+ for longer sessions

For full training runs, dedicated GPU instances (RunPod, Lambda Labs) are often more practical.

## Technical Questions

### What is the `on_filled` function and why doesn't it have the same signature as `on_execution`?

**Answer**: The `on_filled` and `on_execution` functions are part of different event handling systems in the FinGPT trading framework:

- `on_execution`: Triggered when a trade order is executed
- `on_filled`: Triggered when an order is filled (completed)

They have different signatures because they handle different types of events with different parameters:

- `on_execution`: Receives execution details (price, quantity, timestamp)
- `on_filled`: Receives fill confirmation details

This design follows standard trading system patterns where order execution and order fills are distinct events with different information requirements.

## FinGPT-RAG Questions

### Will you open source the fine-tuned FinGPT with RAG that can be used directly?

**Answer**: Currently, we provide the FinGPT-RAG framework and codebase, but not the pre-trained fine-tuned models with RAG integration. This is because:

1. **RAG is modular**: RAG performance depends heavily on the retrieval system and knowledge base
2. **Customization needed**: Optimal RAG configurations vary by use case
3. **Resource considerations**: Full RAG models are large and resource-intensive

However, you can:
1. Use our open-source RAG framework with your own knowledge base
2. Fine-tune the base models using our instruction tuning datasets
3. Build custom RAG systems using our provided components

We are working on providing more complete RAG solutions in future releases.

### Can I use only RAG method to enhance LLM's ability without fine-tuning?

**Answer**: Yes, you can use RAG (Retrieval-Augmented Generation) without fine-tuning the base LLM. This approach:

**Advantages:**
- No need for extensive compute resources for fine-tuning
- Can be implemented with smaller models
- Easier to update knowledge base vs. retraining
- Works with various base LLMs (GPT-3.5, Llama, etc.)

**Limitations:**
- May not achieve the same performance as fine-tuned + RAG
- Dependent on quality of retrieval system
- Base model may not be optimized for financial domain

**Implementation Approach:**
1. Use a general-purpose LLM (GPT-3.5, Llama 2, etc.)
2. Build a financial knowledge base with your data
3. Implement retrieval system (vector database, search engine)
4. Use RAG to augment prompts with relevant context

This approach is particularly useful if you have limited compute resources or need to frequently update your knowledge base.

## Data and Dataset Questions

### Where can I find the datasets used for FinGPT training?

**Answer**: Most FinGPT training datasets are available on HuggingFace:

**Available Datasets:**
- [fingpt-sentiment-train](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train) - Sentiment analysis training data
- [fingpt-finred](https://huggingface.co/datasets/FinGPT/fingpt-finred) - Financial relation extraction
- [fingpt-headline](https://huggingface.co/datasets/FinGPT/fingpt-headline) - Financial headline classification
- [fingpt-ner](https://huggingface.co/datasets/FinGPT/fingpt-ner) - Financial named entity recognition
- [fingpt-fiqa_qa](https://huggingface.co/datasets/FinGPT/fingpt-fiqa_qa) - Financial Q&A
- [fingpt-fineval](https://huggingface.co/datasets/FinGPT/fingpt-fineval) - Chinese multiple-choice questions
- [fingpt-forecaster-dow30-202305-202405](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405) - Forecaster dataset

**For FinGPT-Forecaster specifically:**
The DOW30 dataset used for training FinGPT-Forecaster is available at:
- [fingpt-forecaster-dow30-202305-202405](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405)

This dataset contains:
- Stock price data for DOW 30 companies
- Financial news headlines and content
- Basic financial information
- Time period: May 2023 to May 2024

**For the original DOW dataset or historical data:**
1. Use financial data APIs (yfinance, Alpha Vantage, etc.)
2. Collect data from financial websites
3. Use the data collection scripts provided in the repository
4. The dataset is structured to be easily extensible for other time periods or markets

**Dataset Usage:**
- You can use this dataset to fine-tune other models (Mistral, Llama 3, etc.)
- The dataset format is compatible with standard HuggingFace dataset loading
- You can extend the dataset with newer data following the same structure

### Can I use the datasets to fine-tune other models (e.g., Mistral)?

**Answer**: Yes, our datasets are designed to be model-agnostic and can be used to fine-tune various LLM architectures including Mistral, Llama 2, ChatGLM, etc.

**Steps to fine-tune Mistral with FinGPT datasets:**
1. Download the dataset from HuggingFace
2. Adapt the data format for Mistral's training requirements
3. Use Mistral's fine-tuning scripts or HuggingFace Trainer
4. Adjust hyperparameters for Mistral's architecture

**Benefits of using newer models like Mistral:**
- Often better performance per parameter
- More efficient architectures
- Improved handling of long contexts
- Better multilingual capabilities

We encourage experimentation with different base models!

## Getting Help

If your question isn't answered here:
1. Check the [GitHub Issues](https://github.com/AI4Finance-Foundation/FinGPT/issues)
2. Join our [Discord community](https://discord.gg/trsr8SXpW5)
3. Refer to component-specific READMEs in the `fingpt/` directory
4. Check the [FinGPT documentation](https://ai4finance.org/research/fingpt-open-source-finllm.html)

## Disclaimer

Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.
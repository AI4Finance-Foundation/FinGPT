# Setup Guide: Running FinGPT Locally and on Replit

This guide provides step-by-step instructions for running FinGPT both locally and on Replit, addressing the requirements for different use cases and hardware configurations.

## Table of Contents
- [Hardware Requirements](#hardware-requirements)
- [Local Setup](#local-setup)
- [Replit Setup](#replit-setup)
- [Quick Start Examples](#quick-start-examples)
- [Troubleshooting](#troubleshooting)

## Hardware Requirements

### Minimum Requirements (For Inference Only)
- **CPU**: Any modern multi-core processor
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space
- **GPU**: Not required for cloud API usage, recommended for local models

### Recommended Requirements (For Training/Fine-tuning)
- **CPU**: Modern multi-core processor (Intel i7+/AMD Ryzen 7+)
- **RAM**: 32GB minimum, 64GB recommended
- **Storage**: 50GB+ free space (SSD recommended)
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3090, A100, etc.)
- **CUDA**: 11.8+ for GPU acceleration

### Cloud GPU Options
If you don't have a powerful GPU, consider these cloud platforms:
- **Google Colab**: Free tier with GPU access
- **Kaggle Kernels**: Free GPU access
- **RunPod**: Affordable GPU rentals
- **Vast.ai**: Low-cost GPU marketplace
- **Lambda Labs**: GPU cloud for ML

## Local Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/AI4Finance-Foundation/FinGPT.git
cd FinGPT
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv fingpt_env
source fingpt_env/bin/activate  # On Windows: fingpt_env\Scripts\activate

# Using conda
conda create -n fingpt python=3.8
conda activate fingpt
```

### Step 3: Install Dependencies

#### Basic Installation
```bash
pip install -r requirements.txt
pip install -e .
```

#### For Inference with Local Models
```bash
pip install transformers==4.32.0 peft==0.5.0
pip install sentencepiece accelerate torch
pip install datasets bitsandbytes
# Install triton only for NVIDIA GPUs (Linux/Windows)
pip install triton<2.1.0  # Skip on macOS/M1 or non-NVIDIA systems
```

#### For Training/Fine-tuning
```bash
pip install transformers==4.32.0 peft==0.5.0
pip install sentencepiece accelerate torch
pip install datasets bitsandbytes
# Install triton only for NVIDIA GPUs (Linux/Windows)
pip install triton<2.1.0  # Skip on macOS/M1 or non-NVIDIA systems
pip install deepspeed wandb  # Optional for advanced training
```

#### For FinGPT-Forecaster
```bash
pip install yfinance finnhub-python
pip install gradio beautifulsoup4 requests
```

### Step 4: Verify Installation
```bash
python -c "import transformers; import torch; print('Transformers:', transformers.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Replit Setup

### Step 1: Create a New Replit
1. Go to [replit.com](https://replit.com)
2. Click "Create Repl"
3. Select "Python" as the template
4. Name your repl (e.g., "FinGPT")

### Step 2: Import the Repository
1. In your Replit, click the "Shell" tab
2. Run the following commands:
```bash
git clone https://github.com/AI4Finance-Foundation/FinGPT.git
mv FinGPT/* .
mv FinGPT/.* . 2>/dev/null || true
rmdir FinGPT
```

### Step 3: Configure Replit for FinGPT

#### Update `.replit` file
Create or update the `.replit` file:
```toml
[run]
command = "python main.py"

[env]
PYTHONPATH = "."
```

#### Update `pyproject.toml` (if needed)
Ensure your dependencies are listed:
```toml
[project]
name = "fingpt"
requires-python = ">=3.8"
dependencies = [
    "transformers==4.32.0",
    "peft==0.5.0",
    "torch",
    "accelerate",
    "sentencepiece",
    "datasets",
    "bitsandbytes",
    "numpy",
    "pandas",
]
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
pip install transformers==4.32.0 peft==0.5.0
pip install sentencepiece accelerate torch
pip install datasets bitsandbytes
# Install triton only for NVIDIA GPUs (Linux/Windows)
pip install triton<2.1.0  # Skip on macOS/M1 or non-NVIDIA systems
```

### Step 5: Handle GPU on Replit
Replit offers GPU access on paid plans. To use GPU:
1. Upgrade to a Replit plan with GPU access
2. Enable GPU in your Replit settings
3. The PyTorch installation will automatically detect CUDA

### Step 6: Run FinGPT
```bash
# Run a simple inference script
python -c "from transformers import AutoTokenizer; print('FinGPT ready!')"
```

## Quick Start Examples

### Example 1: Running Inference with Pre-trained Models

#### Using FinGPT-Sentiment Model (Local)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

# Load FinGPT model
model = PeftModel.from_pretrained(
    base_model, 
    'FinGPT/fingpt-sentiment_llama2-13b_lora'
)
model = model.eval()

# Prepare input
text = "Glaxo's ViiV Healthcare Signs China Manufacturing Deal With Desano"
prompt = f"What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.\n\n{text}"

# Generate response
inputs = tokenizer(prompt, return_tensors='pt')
inputs = {key: value.to(model.device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

#### Using Cloud API (No GPU Required)
```python
import os

# Set your API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
os.environ['FINGPT_LLM_PROVIDER'] = 'openai'

# Use FinGPT with OpenAI
from fingpt.Forecaster import FinGPTForecaster

forecaster = FinGPTForecaster()
result = forecaster.predict(ticker="AAPL", date="2024-01-15")
print(result)
```

### Example 2: Running FinGPT-Forecaster Demo

#### Local Setup
```bash
cd fingpt/FinGPT_Forecaster
pip install -r requirements.txt
```

#### Run the demo notebook
```bash
jupyter notebook demo.ipynb
```

Or run the Gradio app:
```python
import gradio as gr
from fingpt.Forecaster import FinGPTForecaster

forecaster = FinGPTForecaster()

def predict(ticker, date, weeks, add_financials):
    result = forecaster.predict(
        ticker=ticker, 
        date=date, 
        weeks=weeks,
        add_financials=add_financials
    )
    return result

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Ticker Symbol"),
        gr.Textbox(label="Date (YYYY-MM-DD)"),
        gr.Slider(1, 12, value=4, label="Number of Weeks"),
        gr.Checkbox(label="Add Basic Financials")
    ],
    outputs="text",
    title="FinGPT-Forecaster"
)

iface.launch()
```

### Example 3: Training with LoRA (Requires GPU)

Use the provided Jupyter notebooks:
- `FinGPT_Training_LoRA_with_ChatGLM2_6B_for_Beginners.ipynb`
- `FinGPT_ Training with LoRA and Meta-Llama-3-8B.ipynb`

```bash
# Start Jupyter
jupyter notebook

# Open and run the training notebook cell by cell
```

## Running Different FinGPT Components

### FinGPT-Sentiment Analysis
```bash
cd fingpt/FinGPT_Sentiment_Analysis_v3
# Run benchmark notebooks
jupyter notebook benchmark/benchmarks.ipynb
```

### FinGPT-Forecaster
```bash
cd fingpt/FinGPT_Forecaster
# Run demo
jupyter notebook demo.ipynb
```

### FinGPT-RAG
```bash
cd fingpt/FinGPT_RAG
# Check the README for specific setup instructions
```

### FinGPT-Benchmark
```bash
cd fingpt/FinGPT_Benchmark
# Run demo
jupyter notebook demo.ipynb
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory
**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Use a smaller model (7B instead of 13B)
- Enable quantization (8-bit or 4-bit)
- Reduce batch size
- Use gradient checkpointing
```python
# Enable 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
```

#### Issue 2: Import Errors
**Problem**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**:
```bash
pip install transformers==4.32.0 peft==0.5.0
pip install sentencepiece accelerate torch
```

#### Issue 3: HuggingFace Authentication
**Problem**: `OSError: meta-llama/Llama-2-7b-chat-hf is a gated model`

**Solution**:
1. Go to [HuggingFace Llama 2 page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
2. Accept the user agreement
3. Generate an access token in your HuggingFace settings
4. Login in your terminal:
```bash
huggingface-cli login
```

#### Issue 4: Replit GPU Not Available
**Problem**: GPU not detected on Replit

**Solution**:
- Upgrade to a Replit plan with GPU access
- Enable GPU in Replit settings
- Use cloud APIs instead of local models

#### Issue 5: Slow Performance on CPU
**Problem**: Inference is very slow on CPU

**Solutions**:
- Use cloud APIs (OpenAI, MiniMax) instead of local models
- Use smaller models
- Enable CPU optimizations:
```python
import torch
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)
```

#### Issue 6: Dependency Conflicts
**Problem**: Version conflicts between packages

**Solution**:
```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

#### Issue 7: Triton Import Error
**Problem**: `ImportError: cannot import name 'driver' from 'triton.runtime'`

**Solution**: This occurs when using incompatible triton versions with bitsandbytes on NVIDIA GPU systems.
```bash
# For NVIDIA GPU systems (Linux/Windows)
pip install triton<2.1.0
# Or downgrade triton if already installed
pip install triton==2.0.1
```

**Note**: On macOS/M1 systems, triton is not required or available. Skip triton installation and use Apple's Metal Performance Shaders (MPS) for GPU acceleration instead.

### Getting Help

If you encounter issues not covered here:
1. Check the [GitHub Issues](https://github.com/AI4Finance-Foundation/FinGPT/issues)
2. Join the [Discord community](https://discord.gg/trsr8SXpW5)
3. Refer to specific component READMEs in the `fingpt/` directory
4. Check the [FinGPT documentation](https://ai4finance.org/research/fingpt-open-source-finllm.html)

## Additional Resources

- [FinGPT Research Paper](https://arxiv.org/abs/2306.06031)
- [HuggingFace Models](https://huggingface.co/FinGPT)
- [FinGPT Demos](https://huggingface.co/spaces/FinGPT)
- [Medium Blog Series](https://medium.datadriveninvestor.com/fingpt-powering-the-future-of-finance-with-20-cutting-edge-applications-7c4d082ad3d8)

## Disclaimer

Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.

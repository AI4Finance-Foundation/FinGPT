![title](figs/title.png)

## What is FinGPT-Forecaster?
- FinGPT-Forecaster takes market news and optional basic financials related to the specified company from the past few weeks as input and responds with the company's **positive developments** and **potential concerns**. Then it gives out a **prediction** of stock price movement for the coming week and its **analysis** summary.
- FinGPT-Forecaster is finetuned on Llama-2-7b-chat-hf with LoRA on the past year's DOW30 market data. But also has shown great generalization ability on other ticker symbols.
- FinGPT-Forecaster is an easy-to-deploy junior robo-advisor, a milestone towards our goal.

## Try out the demo!

Try our demo at <https://huggingface.co/spaces/FinGPT/FinGPT-Forecaster>

![demo_interface](figs/interface.png)

Enter the following inputs:

1) ticker symbol (e.g. AAPL, MSFT, NVDA)
2) the day from which you want the prediction to happen (yyyy-mm-dd)
3) the number of past weeks where market news are retrieved
4) whether to add latest basic financials as additional information

Then, click Submit！You'll get a response like this

![demo_response](figs/response.png)

This is just a demo showing what this model is capable of. Results inferred from randomly chosen news can be strongly biased.
For more detailed and customized usage, scroll down and continue your reading.

## Deploy FinGPT-Forecaster

We have released our FinGPT-Forecaster trained on DOW30 market data from 2022-12-30 to 2023-9-1 on HuggingFace: [fingpt-forecaster_dow30_llama2-7b_lora](https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora)

We have most of the key requirements in `requirements.txt`. Before you start, do `pip install -r requirements.txt`. Then you can refer to `demo.ipynb` for our deployment and evaluation script.

First let's load the model:

```
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,   # optional if you have enough VRAM
)
tokenizer = AutoTokenizer.from_pretrained('FinGPT/fingpt-forecaster_dow30_llama2-7b_lora')

model = PeftModel.from_pretrained('FinGPT/fingpt-forecaster_dow30_llama2-7b_lora')
model = model.eval()
```

Then you are ready to go, prepare your prompt with news & stock price movements in llama format (which we'll mention in the next section), and generate your own forecasting results!
```
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

prompt = B_INST + B_SYS + {SYSTEM_PROMPT} + E_SYS + {YOUR_PROMPT} + E_INST
inputs = tokenizer(
    prompt, return_tensors='pt'
)
inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
res = model.generate(
    **inputs, max_length=4096, do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True
)
output = tokenizer.decode(res[0], skip_special_tokens=True)
answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL) # don't forget to import re
```

## Data Preparation
Company profile & Market news & Basic financials & Stock prices are retrieved using **yfinance & finnhub**.

## Train your own FinGPT-Forecaster



**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**

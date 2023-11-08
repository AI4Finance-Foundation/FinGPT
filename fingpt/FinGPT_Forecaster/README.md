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

We have released our FinGPT-Forecaster trained on DOW30 market data from 2022-12-30 to 2023-9-1 on HuggingFace <https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora>

## Data Preparation
Company profile & Market news & Basic financials & Stock prices are retrieved using **yfinance & finnhub**.

## Train your own FinGPT-Forecaster



**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**

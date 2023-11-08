---
title: FinGPT Forecaster
emoji: 📈
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 4.1.1
app_file: app.py
pinned: false
---

# This is a demo of FinGPT-Forecaster.

FinGPT-Forecaster takes random market news and optional basic financials related to the specified company from the past few weeks as input and responds with the company's **positive developments** and **potential concerns**. Then it gives out a **prediction** of stock price movement for the coming week and its **analysis** summary.

This model is finetuned on Llama2-7b-chat-hf with LoRA on the past year's DOW30 market data. Inference in this demo uses fp16 and **welcomes any ticker symbol**.

Company profile & Market news & Basic financials & Stock prices are retrieved using **yfinance & finnhub**.

This is just a demo showing what this model is capable of. Results inferred from randomly chosen news can be strongly biased.

For more detailed and customized implementation, refer to our **FinGPT project** <https://github.com/AI4Finance-Foundation/FinGPT>

**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**

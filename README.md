# Data-Centric FinGPT: Open-source for Open Finance.
[![Downloads](https://pepy.tech/badge/fingpt)](https://pepy.tech/project/fingpt)
[![Downloads](https://pepy.tech/badge/fingpt/week)](https://pepy.tech/project/fingpt)
[![Python 3.8](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/fingpt.svg)](https://pypi.org/project/fingpt/)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/fingpt.svg?color=brightgreen)

Let us DO NOT expect Wall Street to open-source LLMs nor open APIs, due to FinTech institutes' internal regulations and policies.

We democratize Internet-scale data for financial large language models (FinLLMs) at [FinNLP](https://github.com/AI4Finance-Foundation/FinNLP) and [FinNLP Website](https://ai4finance-foundation.github.io/FinNLP/) 

[Blueprint of FinGPT](https://arxiv.org/abs/2306.06031)

**Disclaimer: We are sharing codes for academic purposes under the MIT education license. Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**

## Why FinGPT?

1). Finance is highly dynamic. [BloombergGPT](https://arxiv.org/abs/2303.17564) retrains an LLM using a mixed dataset of finance and general data sources, which is too expensive (1.3M GPU hours, a cost of around **$5M**). It is costly to retrain an LLM model every month or every week, so lightweight adaptation is highly favorable in finance. Instead of undertaking a costly and time-consuming process of retraining a model from scratch with every significant change in the financial landscape, FinGPT can be fine-tuned swiftly to align with new data (the cost of adaptation falls significantly, estimated at less than **$416 per training**).

2). Democratizing Internet-scale financial data is critical, which should allow timely updates (monthly or weekly updates) using an automatic data curation pipeline. But, BloombergGPT has privileged data access and APIs. FinGPT presents a more accessible alternative. It prioritizes lightweight adaptation, leveraging the strengths of some of the best available open-source LLMs, which are then fed with financial data and fine-tuned for financial language modeling.

3). The key technology is "RLHF (Reinforcement learning from human feedback)", which is missing in BloombergGPT. RLHF enables an LLM model to learn individual preferences (risk-aversion level, investing habits, personalized robo-advisor, etc.), which is the "secret" ingredient of ChatGPT and GPT4.

## FinGPT Demos
* [FinGPT V3 (Updated on 7/11/2023)](./fingpt)
  + **FinGPT v3 [(FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT)](https://huggingface.co/oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT) is a LLM finetuned with LoRA method on the News and Tweets sentiment analysis dataset which achieve best scores on most of the financial sentiment analysis datasets.**
  + Benchmark Results: 
    | Weighted F1   | BloombergGPT | ChatGLM2 | ChatGLM2 (8-bit) | FinGPT v3 | FinGPT v3 (8-bit) |
    | ---------------------- | ------------ | -------- | ---------------- | --------- | ----------------- |
    | FPB  | 0.511        | 0.381    | 0.398            | **0.795** | 0.778             |
    | FiQA-SA   | 0.751        | 0.79     | 0.801            | **0.806** | 0.801             |
    | TFNS   | -            | 0.189    | 0.19             | **0.74**  | 0.721             |
    | NWGI   | - | 0.449    | 0.452            | **0.578** | **0.578**         |

* [FinGPT V2](./fingpt)
  + **Let's train our own FinGPT in American Financial Market with LLaMA and LoRA  (Low-Rank Adaptation)**
* [FinGPT V1](./fingpt)
  + **Let's train our own FinGPT in Chinese Financial Market with ChatGLM and LoRA (Low-Rank Adaptation)**

## Understanding FinGPT: An Educational Blog Series
+ [FinGPT: Powering the Future of Finance with 20 Cutting-Edge Applications
](https://medium.datadriveninvestor.com/fingpt-powering-the-future-of-finance-with-20-cutting-edge-applications-7c4d082ad3d8)
+ [FinGPT I: Why We Built the First Open-Source Large Language Model for Finance
](https://medium.datadriveninvestor.com/fingpt-i-why-we-built-the-first-open-source-large-language-model-for-finance-c01b5517ca)

## What is FinGPT and FinNLP?

### The Goals of FinGPT
1. Real-time data curation pipeline to **democratize data** for FinGPT 
2. Lightweight adaptation to **democratize the FinGPT model** for both individuals and institutes (frequent updates)
3. Support various **financial applications**

* FinNLP provides a playground for all people interested in LLMs and NLP in Finance. Here we provide full pipelines for LLM training and finetuning in the field of finance. The full architecture is shown in the following picture. Detail codes and introductions can be found [here](https://github.com/AI4Finance-Foundation/FinNLP). Or you may refer to the [wiki](https://ai4finance-foundation.github.io/FinNLP/)

<div align="center">
<img align="center" src=figs/FinGPT_framework.png>
</div>

## End-to-end framework: FinGPT embraces a full-stack framework for FinLLMs with four layers:
* **Data source layer**: This layer assures comprehensive market coverage, addressing the temporal sensitivity of financial data through real-time information capture.
* **Data engineering layer**: Primed for real-time NLP data processing, this layer tackles the inherent challenges of high temporal sensitivity and low signal-to-noise ratio in financial data.
* **LLMs layer**: Focusing on a range of fine-tuning methodologies such as LoRA, this layer mitigates the highly dynamic nature of financial data, ensuring the model’s relevance and accuracy.
* **Application layer**: Showcasing practical applications and demos, this layer highlights the potential capability of FinGPT in the financial sector.

## News

+ [Columbia Perspectives on ChatGPT](https://datascience.columbia.edu/news/2023/columbia-perspectives-on-chatgpt/?utm_source=sendinblue&utm_campaign=DSI%20Newsletter%20April%202023&utm_medium=email)
+ [MIT Technology Review] [ChatGPT is about to revolutionize the economy. We need to decide what that looks like](https://www.technologyreview.com/2023/03/25/1070275/chatgpt-revolutionize-economy-decide-what-looks-like/)
+ [BloombergGPT] [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)
+ [Finextra] [ChatGPT and Bing AI to sit as panellists at fintech conference](https://www.finextra.com/newsarticle/41973/chatgpt-and-bing-ai-to-sit-as-panellists-at-fintech-conference)

## ChatGPT at AI4Finance

+ [YouTube video] [I Built a Trading Bot with ChatGPT](https://www.youtube.com/watch?v=fhBw3j_O9LE), combining ChatGPT and FinRL.
+ [Hey, ChatGPT! Explain FinRL code to me!](https://medium.com/@ai4finance/hey-chatgpt-explain-finrl-code-to-me-6a91d612296f)
+ [ChatGPT Robo Advisor v2](./fingpt)
+ [ChatGPT Robo Advisor v1](./demos)
    * A demo of using ChatGPT to build a Robo-advisor 
+ [ChatGPT Trading Agent V2](./fingpt)
    * A FinRL agent that trades as smartly as ChatGPT by using the large language model behind ChatGPT
+ [ChatGPT Trading Agent V1](./fingpt)
    * Trade with the suggestions given by ChatGPT
+ ChatGPT adds technical indicators into FinRL

## Introductory

+ [Sparks of artificial general intelligence: Early experiments with GPT-4](https://arxiv.org/abs/2303.12712)
+ [GPT-4] [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
+ [InstructGPT] [Training language models to follow instructions with human feedback](https://openreview.net/forum?id=TG8KACxEON) NeurIPS 2022.

[The Journey of Open AI GPT models](https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2).  GPT models explained. Open AI's GPT-1, GPT-2, GPT-3.

+ [GPT-3] [Language models are few-shot learners](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) NeurIPS 2020.
+ [GPT-2] [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
+ [GPT-1] [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
+ [Transformer] [Attention is All you Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) NeurIPS 2017.

## (Financial) Big Data

+ [BloombergGPT] [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)

+ [WHAT’S IN MY AI?](https://lifearchitect.ai/whats-in-my-ai/) A Comprehensive Analysis of Datasets Used to Train GPT-1, GPT-2, GPT-3, GPT-NeoX-20B, Megatron-11B, MT-NLG, and Gopher

+ [FinRL-Meta Repo](https://github.com/AI4Finance-Foundation/FinRL-Meta) and paper [FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0bf54b80686d2c4dc0808c2e98d430f7-Abstract-Datasets_and_Benchmarks.html). Advances in Neural Information Processing Systems, 2022.

+ [AI4Finance] [FinNLP](https://github.com/AI4Finance-Foundation/FinNLP) Democratizing Internet-scale financial data.

## Interesting Demos

+ [GPT-3 Creative Fiction](https://gwern.net/gpt-3#prompts-as-programming) Creative writing by OpenAI’s GPT-3 model, demonstrating poetry, dialogue, puns, literary parodies, and storytelling. Plus advice on effective GPT-3 prompt programming & avoiding common errors.

## ChatGPT for FinTech

**ChatGPT Trading Bot**
+ [YouTube video] [ChatGPT Trading strategy 20097% returns](https://www.youtube.com/watch?v=unsa_gXPAJ4)
+ [YouTube video] [ChatGPT Coding - Make A Profitable Trading Strategy In Five Minutes!](https://www.youtube.com/watch?v=4SG2884RcDY)
+ [YouTube video] [Easy Automated Live Trading using ChatGPT (+9660.3% hands free)](https://www.youtube.com/watch?v=dIEZVPVOZPQ)
+ [YouTube video] [ChatGPT Trading Strategy 893% Returns](https://www.youtube.com/watch?v=YxjvjK5AD2M)
+ [YouTube video] [ChatGPT 10 Million Trading Strategy](https://www.youtube.com/watch?v=9VPfd08uU4Q)
+ [YouTube video] [ChatGPT: Your Crypto Assistant](https://www.youtube.com/watch?v=LpzeshX6s2w)
+ [YouTube video] [Generate Insane Trading Returns with ChatGPT and TradingView](https://www.youtube.com/watch?v=ekz6ugJE1h0&t=3s)


**(Fast and accurate) Sentiment Analysis**

   GPT-3 can help study customer surveys, social media tweets from customers/users.

   Tweets
+ [Tweet Classifier](https://platform.openai.com/playground/p/default-tweet-classifier?model=text-davinci-003)
+ [Advanced Tweet Classifier](https://platform.openai.com/playground/p/default-adv-tweet-classifier?model=text-davinci-003)

  Financial News
+ [Algorithmic Trading using Sentiment Analysis on News Articles](https://towardsdatascience.com/https-towardsdatascience-com-algorithmic-trading-using-sentiment-analysis-on-news-articles-83db77966704)
+ [Accessing Historical Financial News Headlines with Python](https://python.plainenglish.io/access-historical-financial-news-headlines-with-python-be1b8faaea9f)

**PromptNet** Analogy to ImageNet and WordNet, it is critical to build a PromptNet.

+ [Awesome_Prompting_Papers_in_Computer_Vision](https://github.com/ttengwang/Awesome_Prompting_Papers_in_Computer_Vision)
+ [OpenPrompt](https://github.com/thunlp/OpenPrompt)
+ [promptsource](https://github.com/bigscience-workshop/promptsource)

**Robo-advisor**

**Coding-tutor**

+ [Hey, ChatGPT! Explain FinRL code to me!](https://medium.com/@ai4finance/hey-chatgpt-explain-finrl-code-to-me-6a91d612296f)

**Blogs about ChatGPT for FinTech**

## ChatGPT APIs

Prompting as a new programming paradigm!
+ [Towards Data Science] [GPT-3: Creative Potential of NLP](https://towardsdatascience.com/gpt-3-creative-potential-of-nlp-d5ccae16c1ab)
+ [YouTube video] [OpenAI GPT-3 - Prompt Engineering For Financial NLP](https://www.youtube.com/watch?v=Nl2Cdbao5Ws)

+ [OpenAI API for GPT-3](https://platform.openai.com/docs/models/gpt-3)
+ [ChatGPT-wrapper: python and shell](https://github.com/mmabrouk/chatgpt-wrapper)
+ [OpenAI Examples Library](https://platform.openai.com/examples)
+ [GPT-3 Sandbox (Github)](https://github.com/shreyashankar/gpt3-sandbox) Enable users to create cool web demos using OpenAI GPT-3 API.
+ [Exploring the Capabilities of the ChatGPT API: A Beginner’s Guide](https://levelup.gitconnected.com/exploring-the-capabilities-of-the-chatgpt-api-a-beginners-guide-e9089d49961f)
+ [Reverse engineered ChatGPT API](https://github.com/acheong08/ChatGPT)

**Prompting programming**

## ChatGPT relatives: 

[A Release Timeline](https://github.com/osanseviero/ml_timeline) of many LLMs.

[PaLM](https://arxiv.org/abs/2204.02311)

[Chincella](https://arxiv.org/abs/2203.15556)

Interesting evaluations:
+ [RLHF for pretraining](https://arxiv.org/abs/2302.08582)

+ [Compare ChatGPT with GPT3.5](https://arxiv.org/pdf/2302.06476.pdf)

+ [Is ChatGPT A Good Translator? A Preliminary Study](https://arxiv.org/pdf/2301.08745.pdf)

+ [A Multitask, Multilingual, Multimodal Evaluation of ChatGPT
on Reasoning, Hallucination, and Interactivity](https://arxiv.org/pdf/2302.04023.pdf)

[YouTube video] [Physics Solution: ChatGPT vs. Google](https://www.youtube.com/watch?v=x4dIx9VYQoM)

## Links

+ [LLM Survey](https://github.com/RUCAIBox/LLMSurvey)
+ [Awesome GPT-3 Examples](https://github.com/elyase/awesome-gpt3)

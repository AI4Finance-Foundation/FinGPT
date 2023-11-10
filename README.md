# FinGPT: Open-Source Financial Large Language Models
[![Downloads](https://static.pepy.tech/badge/fingpt)](https://pepy.tech/project/fingpt)
[![Downloads](https://static.pepy.tech/badge/fingpt/week)](https://pepy.tech/project/fingpt)
[![Python 3.8](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/fingpt.svg)](https://pypi.org/project/fingpt/)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/fingpt.svg?color=brightgreen)


<div align="center">
<img align="center" src=figs/logo_transparent_background.png width="40%"/>
</div>

Let us not expect Wall Street to open-source LLMs or open APIs, due to FinTech institutes' internal regulations and policies.

[Blueprint of FinGPT](https://arxiv.org/abs/2306.06031)

[![](https://dcbadge.vercel.app/api/server/trsr8SXpW5)](https://discord.gg/trsr8SXpW5)

## What's New:
 - [Model Release] Nov, 2023: We release [FinGPT-Forecaster](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Forecaster)! 😊 [Demo](https://huggingface.co/spaces/FinGPT/FinGPT-Forecaster) & [Model](https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora) are available on Huggingface!
 - [Paper Acceptance] Oct, 2023: ["FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets"]() is accepted🎉 by Instruction Workshop @ NeurIPS 2023 
 - [Paper Acceptance] Oct, 2023: ["FinGPT: Democratizing Internet-scale Data for Financial Large Language Models"]() is accepted🎉 by Instruction Workshop @ NeurIPS 2023
 - [Model Release] Oct, 2023: We release the [financial multi-task LLMs](https://huggingface.co/FinGPT) 🔥 produced when evaluating base-LLMs on [FinGPT-Benchmark](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Benchmark)
 - [Paper Acceptance] Sep, 2023: ["Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General-Purpose Large Language Models"](https://arxiv.org/abs/2306.12659) ...
 - [Model Release] ...

## Why FinGPT?

1). Finance is highly dynamic. [BloombergGPT](https://arxiv.org/abs/2303.17564) trained an LLM using a mixture of finance data and general-purpose data, which took about 53 days, at a cost of around **$3M**). It is costly to retrain an LLM model like BloombergGPT every month or every week, thus lightweight adaptation is highly favorable. FinGPT can be fine-tuned swiftly to incorporate new data (the cost falls significantly, less than **$300 per fine-tuning**).

2). Democratizing Internet-scale financial data is critical, say allowing timely updates of the model (monthly or weekly updates) using an automatic data curation pipeline.  BloombergGPT has privileged data access and APIs, while FinGPT presents a more accessible alternative. It prioritizes lightweight adaptation, leveraging the best available open-source LLMs.

3). The key technology is "RLHF (Reinforcement learning from human feedback)", which is missing in BloombergGPT. RLHF enables an LLM model to learn individual preferences (risk-aversion level, investing habits, personalized robo-advisor, etc.), which is the "secret" ingredient of ChatGPT and GPT4.

## Instruction Tuning Datasets and Models
The datasets we used, and the **multi-task financial LLM** models are available at <https://huggingface.co/FinGPT>

[Our Code](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Benchmark)
  
  | Datasets | Train Rows |  Test Rows |Description  |
  | --------- | ----------------- | ------------ | --------------------- |
  | [fingpt-sentiment-train](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train) | 76.8K | N/A|Sentiment Analysis Training Instructions |
  | [fingpt-finred](https://huggingface.co/datasets/FinGPT/fingpt-finred)| 27.6k | 5.11k | Financial Relation Extraction Instructions |
  | [fingpt-headline](https://huggingface.co/datasets/FinGPT/fingpt-headline) | 82.2k | 20.5k | Financial Headline Analysis Instructions|
  | [fingpt-ner](https://huggingface.co/datasets/FinGPT/fingpt-ner) | 511   | 98  | Financial Named-Entity Recognition Instructions|
  | [fingpt-fiqa_qa](https://huggingface.co/datasets/FinGPT/fingpt-fiqa_qa) | 17.1k   | N/A  | Financial Q&A Instructions|
  | [fingpt-fineval](https://huggingface.co/datasets/FinGPT/fingpt-fineval) | 1.06k   | 265  | Chinese Multiple-Choice Questions Instructions|

  Multi-task financial LLMs Models:
```python
  demo_tasks = [
      'Financial Sentiment Analysis',
      'Financial Relation Extraction',
      'Financial Headline Classification',
      'Financial Named Entity Recognition',]
  demo_inputs = [
      "Glaxo's ViiV Healthcare Signs China Manufacturing Deal With Desano",
      "Apple Inc. Chief Executive Steve Jobs sought to soothe investor concerns about his health on Monday, saying his weight loss was caused by a hormone imbalance that is relatively simple to treat.",
      'gold trades in red in early trade; eyes near-term range at rs 28,300-28,600',
      'This LOAN AND SECURITY AGREEMENT dated January 27 , 1999 , between SILICON VALLEY BANK (" Bank "), a California - chartered bank with its principal place of business at 3003 Tasman Drive , Santa Clara , California 95054 with a loan production office located at 40 William St ., Ste .',]
  demo_instructions = [
      'What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.',
      'Given phrases that describe the relationship between two words/phrases as options, extract the word/phrase pair and the corresponding lexical relationship between them from the input text. The output format should be "relation1: word1, word2; relation2: word3, word4". Options: product/material produced, manufacturer, distributed by, industry, position held, original broadcaster, owned by, founded by, distribution format, headquarters location, stock exchange, currency, parent organization, chief executive officer, director/manager, owner of, operator, member of, employer, chairperson, platform, subsidiary, legal form, publisher, developer, brand, business division, location of formation, creator.',
      'Does the news headline talk about price going up? Please choose an answer from {Yes/No}.',
      'Please extract entities and their types from the input sentence, entity types should be chosen from {person/organization/location}.',]
```

  | Models | Description  | Function |
  | --------- | --------------------- |---------------- |
  | [fingpt-mt_llama2-7b_lora](https://huggingface.co/FinGPT/fingpt-mt_llama2-7b_lora)| Fine-tuned Llama2-7b model with LoRA | Multi-Task |
  | [fingpt-mt_falcon-7b_lora](https://huggingface.co/FinGPT/fingpt-mt_falcon-7b_lora)| Fine-tuned falcon-7b model with LoRA  | Multi-Task |
  | [fingpt-mt_bloom-7b1_lora](https://huggingface.co/FinGPT/fingpt-mt_bloom-7b1_lora) | Fine-tuned bloom-7b1 model with LoRA | Multi-Task |
  | [fingpt-mt_mpt-7b_lora](https://huggingface.co/FinGPT/fingpt-mt_mpt-7b_lora) | Fine-tuned mpt-7b model with LoRA | Multi-Task |
  | [fingpt-mt_chatglm2-6b_lora](https://huggingface.co/FinGPT/fingpt-mt_chatglm2-6b_lora) | Fine-tuned chatglm-6b model with LoRA | Multi-Task |
  | [fingpt-mt_qwen-7b_lora](https://huggingface.co/FinGPT/fingpt-mt_qwen-7b_lora) | Fine-tuned qwen-7b model with LoRA | Multi-Task |
  | [fingpt-sentiment_llama2-13b_lora](https://huggingface.co/FinGPT/fingpt-sentiment_llama2-13b_lora) | Fine-tuned llama2-13b model with LoRA | Single-Task |
  | [fingpt-forecaster_dow30_llama2-7b_lora](https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora) | Fine-tuned llama2-7b model with LoRA | Single-Task |


## FinGPT Demos: Current State-of-the-arts for Financial Sentiment Analysis
* [FinGPT V3 (Updated on 10/12/2023)](./fingpt)
  
  * What's new: **Best trainable and inferable FinGPT for sentiment analysis on a single RTX 3090, which is even better than GPT-4 and ChatGPT Finetuning.**
  
  * [FinGPT v3](https://huggingface.co/FinGPT/fingpt-sentiment_llama2-13b_lora) series are LLMs finetuned with the LoRA method on the News and Tweets sentiment analysis dataset which achieve the best scores on most of the financial sentiment analysis datasets with low cost.
  
  * FinGPT v3.3 use llama2-13b as base model; FinGPT v3.2 uses llama2-7b as base model; FinGPT v3.1 uses chatglm2-6B as base model.
  
  * Benchmark Results:
  
  * | Weighted F1                                                  |    FPB    |  FiQA-SA  |   TFNS    |   NWGI    |      Devices       |    Time     |      Cost      |
    | ------------------------------------------------------------ | :-------: | :-------: | :-------: | :-------: | :----------------: | :---------: | :------------: |
    | [FinGPT v3.3](https://huggingface.co/FinGPT/fingpt-sentiment_llama2-13b_lora)| **0.882** |   0.874   | **0.903** | **0.643** |    1 × RTX 3090    | 17.25 hours |     $17.25     |
    | FinGPT v3.2|   0.850   |   0.860   |   0.894   |   0.636   |      1 × A100      |  5.5 hours  |    $ 22.55     |
    | FinGPT v3.1|   0.855   |   0.850   |   0.875   |   0.642   |      1 × A100      |  5.5 hours  |    $ 22.55     |
    | FinGPT (8bit)                                                |   0.855   |   0.847   |   0.879   |   0.632   |    1 × RTX 3090    | 6.47 hours  |     $ 6.47     |
    | FinGPT (QLoRA)                                               |   0.777   |   0.752   |   0.828   |   0.583   |    1 × RTX 3090    | 4.15 hours  |     $ 4.15     |
    | OpenAI Fine-tune                                             |   0.878   | **0.887** |   0.883   |     -     |         -          |      -      |       -        |
    | GPT-4                                                        |   0.833   |   0.630   |   0.808   |     -     |         -          |      -      |       -        |
    | FinBERT                                                      |   0.880   |   0.596   |   0.733   |   0.538   | 4 × NVIDIA K80 GPU |      -      |       -        |
    | Llama2-7B                                                    |   0.390   |   0.800   |   0.296   |   0.503   |    2048 × A100     |   21 days   | $ 4.23 million |
    | BloombergGPT                                                 |   0.511   |   0.751   |     -     |     -     |     512 × A100     |   53 days   | $ 2.67 million |
  
    **Cost per GPU hour.** For **A100 GPUs**, the AWS p4d.24xlarge instance, equipped with 8 A100 GPUs is used as a benchmark to estimate the costs. Note that BloombergGPT also used p4d.24xlarge As of July 11, 2023, the hourly rate for this instance stands at $32.773. Consequently, the estimated cost per GPU hour comes to $32.77 divided by 8, resulting in approximately **$4.10**. With this value as the reference unit price (1 GPU hour). **BloombergGPT estimated cost= 512 x 53 x 24 = 651,264 GPU hours x $4.10 = $2,670,182.40**. For **RTX 3090**, we assume its cost per hour is approximately **$1.0**, which is actually much higher than available GPUs from platforms like vast.ai.
  
  * Reproduce the results by running [benchmarks](./fingpt/FinGPT_v3/benchmark/benchmarks.ipynb), and the detailed tutorial is on the way.
  * Finetune your own FinGPT v3 model with the LoRA method on only an RTX 3090 with this [notebook](./fingpt/FinGPT_v3/training_8bit/train_Llama2_13B.ipynb) in 8bit or this [notebook](./fingpt/FinGPT_v3/training_int4/train.ipynb) in int4 (QLoRA)
  
* [FinGPT V1](./fingpt)
  + **FinGPT by finetuning ChatGLM2 / Llama2 with LoRA with the market-labeled data for the Chinese Market**

  
## Tutorials
[[Training] Beginner’s Guide to FinGPT: Training with LoRA and ChatGLM2–6B One Notebook, $10 GPU](https://byfintech.medium.com/beginners-guide-to-fingpt-training-with-lora-chatglm2-6b-9eb5ace7fe99)

## Understanding FinGPT: An Educational Blog Series
+ [FinGPT: Powering the Future of Finance with 20 Cutting-Edge Applications
](https://medium.datadriveninvestor.com/fingpt-powering-the-future-of-finance-with-20-cutting-edge-applications-7c4d082ad3d8)
+ [FinGPT I: Why We Built the First Open-Source Large Language Model for Finance
](https://medium.datadriveninvestor.com/fingpt-i-why-we-built-the-first-open-source-large-language-model-for-finance-c01b5517ca)
+ [FinGPT II: Cracking the Financial Sentiment Analysis Task Using Instruction Tuning of General-Purpose Large Language Models
](https://medium.datadriveninvestor.com/fingpt-ii-cracking-the-financial-sentiment-analysis-task-using-instruction-tuning-of-3333bce428c4)


## FinGPT Ecosystem
### FinGPT embraces a full-stack framework for FinLLMs with five layers:
1. **Data source layer**: This layer assures comprehensive market coverage, addressing the temporal sensitivity of financial data through real-time information capture.
2. **Data engineering layer**: Primed for real-time NLP data processing, this layer tackles the inherent challenges of high temporal sensitivity and low signal-to-noise ratio in financial data.
3. **LLMs layer**: Focusing on a range of fine-tuning methodologies such as LoRA, this layer mitigates the highly dynamic nature of financial data, ensuring the model’s relevance and accuracy.
4. **Task layer**: This layer is responsible for executing fundamental tasks. These tasks serve as the benchmarks for performance evaluations and cross-comparisons in the realm of FinLLMs
5. **Application layer**: Showcasing practical applications and demos, this layer highlights the potential capability of FinGPT in the financial sector.

* FinGPT Framework: Open-Source Financial Large Language Models

<div align="center">
<img align="center" src=figs/FinGPT_framework_20231003.png>
</div>

* [FinGPT-RAG](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_RAG): We present a retrieval-augmented large language model framework specifically designed for financial sentiment analysis, optimizing information depth and context through external knowledge retrieval, thereby ensuring nuanced predictions.

<div align="center">
<img align="center" src=figs/FinGPT_RAG_framework.png>
</div>

* [FinGPT-FinNLP](https://github.com/AI4Finance-Foundation/FinNLP): FinNLP provides a playground for all people interested in LLMs and NLP in Finance. Here we provide full pipelines for LLM training and finetuning in the field of finance. The full architecture is shown in the following picture. Detail codes and introductions can be found [here](https://github.com/AI4Finance-Foundation/FinNLP). Or you may refer to the [wiki](https://ai4finance-foundation.github.io/FinNLP/)

<div align="center">
<img align="center" src=figs/FinGPT_FinNLP_data_source.png>
</div>

* [FinGPT-Benchmark](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Benchmark): We introduce a novel Instruction Tuning paradigm optimized for open-source Large Language Models (LLMs) in finance, enhancing their adaptability to diverse financial datasets while also facilitating cost-effective, systematic benchmarking from task-specific, multi-task, and zero-shot instruction tuning tasks. 


<div align="center">
<img align="center" src=figs/FinGPT_Benchmark_update1.png>
</div>



## Open-Source Base Model used in the LLMs layer of FinGPT
* Feel free to contribute more open-source base models tailored for various language-specific financial markets.

| Base Model |Pretraining Tokens|Context Length  | Model Advantages |Model Size|Experiment Results |  Applications |
|  ----  |  ----  |  ----  |   ----  |   ----  |  ----  | ----  |
| [Llama-2](https://github.com/facebookresearch/llama)|2 Trillion|4096| Llama-2 excels on English-based market data | [llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) and [Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf) | llama-2 consistently shows superior fine-tuning results  | Financial Sentiment Analysis, Robo-Advisor |
| [Falcon](https://github.com/falconry/falcon) |1,500B|2048|  Maintains high-quality results while being more resource-efficient | [falcon-7b](https://huggingface.co/tiiuae/falcon-7b) |Good for English market data  | Financial Sentiment Analysis |
| [MPT](https://github.com/mosaicml/llm-foundry) |1T|2048| MPT models can be trained with high throughput efficiency and stable convergence | [mpt-7b](https://huggingface.co/mosaicml/mpt-7b) |Good for English market data  | Financial Sentiment Analysis |
| [Bloom](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml#readme) |366B|2048| World’s largest open multilingual language model  | [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1) |Good for English market data  | Financial Sentiment Analysis |
| [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)|1.4T  |32K |Exceptional capability for Chinese language expression| [chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b) |Shows prowess for Chinese market data  | Financial Sentiment Analysis, Financial Report Summary |
| [Qwen](https://github.com/QwenLM/Qwen-7B)|2.2T  |8k |Fast response and high accuracy| [qwen-7b](https://huggingface.co/tangger/Qwen-7B-Chat) |Effective for Chinese market data  | Financial Sentiment Analysis|
| [InternLM](https://github.com/InternLM/InternLM) |1.8T  |8k |Can flexibly and independently construct workflows |[internlm-7b](https://huggingface.co/internlm/internlm-7b) |Effective for Chinese market data  | Financial Sentiment Analysis |

* Benchmark Results for the above open-source Base Models in the financial sentiment analysis task using the same instruction template for SFT (LoRA):
  | Weighted F1/Acc  |Llama2 |Falcon |  MPT|Bloom |ChatGLM2|Qwen|InternLM |
  | --------- | ----------------- | ------------ | --------------------- | ---------------- | --------------- | ----------------- |----------------- |
  | [FPB](https://huggingface.co/datasets/financial_phrasebank) | 0.863/0.863 | 0.846/0.849  | **0.872**/**0.872**   | 0.810/0.810 | 0.850/0.849 |0.854/0.854| 0.709/0.714 |
  | [FiQA-SA](https://huggingface.co/datasets/pauri32/fiqa-2018)| **0.871**/0.855| 0.840/0.811  | 0.863/0.844 | 0.771/0.753| 0.864/**0.862** | 0.867/0.851  |0.679/0.687 |
  | [TFNS](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | 0.896/0.895 | 0.893/0.893 | **0.907**/**0.907** | 0.840/0.840 | 0.859/0.858 | 0.883/0.882|0.729/0.731|
  | [NWGI](https://huggingface.co/datasets/oliverwang15/news_with_gpt_instructions) | **0.649/0.651**   | 0.636/0.638  | 0.640/0.641| 0.573/0.574| 0.619/0.629 |0.638/0.643|0.498/0.503|

### All Thanks To Our Contributors :
<a href="https://github.com/AI4Finance-Foundation/FinGPT/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Finance-Foundation/FinGPT" />
</a>

## News

+ [Columbia Perspectives on ChatGPT](https://datascience.columbia.edu/news/2023/columbia-perspectives-on-chatgpt/?utm_source=sendinblue&utm_campaign=DSI%20Newsletter%20April%202023&utm_medium=email)
+ [MIT Technology Review] [ChatGPT is about to revolutionize the economy. We need to decide what that looks like](https://www.technologyreview.com/2023/03/25/1070275/chatgpt-revolutionize-economy-decide-what-looks-like/)
+ [BloombergGPT] [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)
+ [Finextra] [ChatGPT and Bing AI to sit as panellists at fintech conference](https://www.finextra.com/newsarticle/41973/chatgpt-and-bing-ai-to-sit-as-panellists-at-fintech-conference)

## ChatGPT at AI4Finance

+ [YouTube video] [I Built a Trading Bot with ChatGPT](https://www.youtube.com/watch?v=fhBw3j_O9LE), combining ChatGPT and FinRL.
+ [Hey, ChatGPT! Explain FinRL code to me!](https://medium.com/@ai4finance/hey-chatgpt-explain-finrl-code-to-me-6a91d612296f)

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

<!--- 
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
---> 

## Citing FinGPT
```
@article{yang2023fingpt,
  title={FinGPT: Open-Source Financial Large Language Models},
  author={Yang, Hongyang and Liu, Xiao-Yang and Wang, Christina Dan},
  journal={FinLLM Symposium at IJCAI 2023},
  year={2023}
}
@article{zhang2023instructfingpt,
      title={Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General-Purpose Large Language Models}, 
      author={Boyu Zhang and Hongyang Yang and Xiao-Yang Liu},
      journal={FinLLM Symposium at IJCAI 2023},
      year={2023}
}
@article{zhang2023fingptrag,
  title={Enhancing Financial Sentiment Analysis via Retrieval Augmented Large Language Models},
  author={Zhang, Boyu and Yang, Hongyang and Zhou, tianyu and Babar, Ali and Liu, Xiao-Yang},
 journal = {ACM International Conference on AI in Finance (ICAIF)},
  year={2023}
}

@article{wang2023fingptbenchmark,
  title={FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets},
  author={Wang, Neng and Yang, Hongyang and Wang, Christina Dan},
  journal={NeurIPS Workshop on Instruction Tuning and Instruction Following},
  year={2023}
}
```

<div align="center">
<a href="https://finllm.github.io/workshop/#/fcb" target="_blank">
<img align="center" src=figs/fingpt_best_presentation.png width="65%">
</div>


## LICENSE

MIT License

**Disclaimer: We are sharing codes for academic purposes under the MIT education license. Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**


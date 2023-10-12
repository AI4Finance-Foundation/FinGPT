## FinGPT Sentiment Analysis

## Motivations:
* Classify financial statements to help traders aggregate and digest financial news
## Methods:
* FinGPT fine-tuning
* Retrieval-Augmented Generation


## Setup

* Visit environment_news_scraping.yml for the environment setup
* Set up your .env file, can refer to /FinGPT_sentiment/.env.example

``` python

python news_scraper.py

```

## I. Data Preparation
Task 1: GPT-based News Classification

1. On UI, select csv file to load
2. Creates "classification" column for each financial statement
3. Using "default_classification_prompt" to ask GPT to classify the news
4. Saves .csv

Task 2: Context Retrieval

1. On UI, select csv file to load
2. Creates "contextualized_sentence" for each financial statement
3. Using Google and various news sources to retrieve the context
   4. Add relevant news paragraphs to form "contextualized_sentence" for each financial statement
5. Saves .csv

## Experiment:compare non-RAG vs. RAG sentiment classification using gpt:
1. utils/sentiment_classification_by_external_LLMs.py: Call openAI APIs to classify RAG-based and non-RAG-based statements
2. utils/classification_accuracy_verification.py: Calculate the accuracy of sentiment classification between RAG-based and non-RAG-based statements
3. Results: 0.7876588021778584 vs. 0.8130671506352088
``` python

python utils/sentiment_classification_by_external_LLMs.py
python utils/classification_accuracy_verification.py

```

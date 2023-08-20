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

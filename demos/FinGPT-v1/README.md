# FinGPT - v1
## Let's train our own ChatGPT in Finance with pre-trained LLMs and LoRA

### Ⅰ. Data Preparations
#### 1. [Download Titles](./data_preparations/download_titles.py)
* In this file, we downloaded the financial news titles and URLs from [eastmoney(东方财富)](https://www.eastmoney.com/)  
#### 2. [Download Content](./data_preparations/download_contents.py)
* In this file, we downloaded the financial news Contents from [eastmoney(东方财富)](https://www.eastmoney.com/)  
#### 2. [Add labels](./data_preparations/add_labels.py)
* In this file, we add the label for news titles and contents.
* The labels are determined by the change pct between the stock price of today and 5-days later
    * change pct >= 0.06 : `very positive` 
    * 0.02 <= change pct <= 0.06 : `positive`
    * -0.02 <= change pct <= 0.02 : `neutral`
    * -0.06 <= change pct <= -0.02 : `negative` 
    * change pct <= -0.06 : `very negative` 

### Ⅱ. Making Dataset
* TO BE CONTINUED

### Ⅲ. Training (Finetuning)
* TO BE CONTINUED

### Ⅳ. Inferencing 
* TO BE CONTINUED

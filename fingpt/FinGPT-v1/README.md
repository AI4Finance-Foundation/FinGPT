# FinGPT-V1 (Labeled by the Market)
## Let's obtain our own FinGPT by finetuning ChatGLM2 / Llama2 with LoRA with the market-labeled data

### Ⅰ. Data Preparations
#### 1. Download Titles [code](./data_preparations/download_titles.py)
* In this file, we downloaded the financial news titles and URLs from [eastmoney(东方财富)](https://www.eastmoney.com/)  
#### 2. Download Content [code](./data_preparations/download_contents.py)
* In this file, we downloaded the financial news Contents from [eastmoney(东方财富)](https://www.eastmoney.com/)  
#### 3. Add labels [code](./data_preparations/add_labels.py)
* In this file, we add the label for news titles and contents.
* The labels are determined by the change pct between the stock price of today and 5-days later
    * change pct >= 0.06 : `very positive` 
    * 0.02 <= change pct <= 0.06 : `positive`
    * -0.02 <= change pct <= 0.02 : `neutral`
    * -0.06 <= change pct <= -0.02 : `negative` 
    * change pct <= -0.06 : `very negative` 

### Ⅱ. Making Dataset

1. Make dataset_by_date [code](https://github.com/AI4Finance-Foundation/FinGPT/blob/master/fingpt/FinGPT-v1/making_dataset/make_dataset_by_date.ipynb)
   * You may run this notebook to generate the dataset file in alpaca format

2. Please run the following two files respectively to generate the dataset in hugging face dataset format.

   * [change_jsonl_train_and_valid.sh](./making_dataset/change_jsonl_train_and_valid.sh)

   * [make_dataset_train_and_valid.sh](./making_dataset/make_dataset_train_and_valid.sh)

### Ⅲ. Training (Finetuning)
* Please run the following codes
    ``` shell
    cd training
    sh finetune.sh
    ```

### Ⅳ. Inferencing 
* Please refer to [infer.ipynb](./inferencing/infer.ipynb)

### Special thanks to [ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning) for introductions on how to finetune ChatGLM by using huggingface.

# FinGPT - v1 (Chinese Financial News + ChatGLM + LoRA)
## Let's train our own ChatGPT in Finance with pre-trained LLMs and LoRA
### Special thanks to [ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning) for introduction on how to finetune ChatGLM by using hugging face.

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

1. [Make dataset_by_date](./making_dataset/make_dataset_by_date.py)
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

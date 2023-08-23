# FinGPT-v2
## Let's obtain our own FinGPT by finetuning Llama2 with LoRA

### Ⅰ. Data Preparations
#### 1. Download Titles [code](./data_preparations/FMP.ipynb)
* In this file, we downloaded the financial news titles and URLs from [Financial Modeling Prep](https://site.financialmodelingprep.com/) 
* You may need to apply for your api key [here](https://site.financialmodelingprep.com/developer/docs/api-keys/)  

#### 2. Add labels
* In this file, we add the label for news titles and contents.
* The labels are determined by the change pct between the stock price of today and 5-days later
    * change pct >= 0.06 : `very positive` 
    * 0.02 <= change pct <= 0.06 : `positive`
    * -0.02 <= change pct <= 0.02 : `neutral`
    * -0.06 <= change pct <= -0.02 : `negative` 
    * change pct <= -0.06 : `very negative` 

### Ⅱ. Making Dataset
* Almost the same as FinGPT-V1 

### Ⅲ. Training (Finetuning)
* Almost the same as FinGPT-V1 

### Ⅳ. Inferencing
* Almost the same as FinGPT-V1 
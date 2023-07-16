# FinGPT v3 Series

**FinGPT v3 series are LLMs finetuned with LoRA method on the News and Tweets sentiment analysis dataset which achieve best scores on most of the financial sentiment analysis datasets.**

## Ⅰ. Try our model ( [FinGPT v3](https://huggingface.co/oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT) )

``` python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Load Models
base_model = "THUDM/chatglm2-6b"
peft_model = "oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModel.from_pretrained(base_model, trust_remote_code=True,  device_map = "auto")
model = PeftModel.from_pretrained(model, peft_model)
model = model.eval()

# Make prompts
prompt = [
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: FINANCING OF ASPOCOMP 'S GROWTH Aspocomp is aggressively pursuing its growth strategy by increasingly focusing on technologically more demanding HDI printed circuit boards PCBs .
Answer: ''',
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .
Answer: ''',
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: A tinyurl link takes users to a scamming site promising that users can earn thousands of dollars by becoming a Google ( NASDAQ : GOOG ) Cash advertiser .
Answer: ''',
]

# Generate results
tokens = tokenizer(prompt, return_tensors='pt', padding=True, max_length=512)
res = model.generate(**tokens, max_length=512)
res_sentences = [tokenizer.decode(i) for i in res]
out_text = [o.split("Answer: ")[1] for o in res_sentences]

# show results
for sentiment in out_text:
    print(sentiment)

# Output:    
# positive
# neutral
# negative
```

## Ⅱ. Benchmark Results

| ACC/F1 Micro    | BloombergGPT | [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B) | [FinGPT v3](https://huggingface.co/oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT) | FinGPT v3.1      | FinGPT v3.2 (8bit) | FinGPT v3.3 (4-bit, QLoRA) |
| --------------- | ------------ | ------------------------------------------------ | ------------------------------------------------------------ | ---------------- | ------------------ | -------------------------- |
| FPB [1]         | -            | 0.464                                            | 0.800                                                        | 0.775            | **0.851**          | <u>0.832</u>               |
| FiQA-SA [2]     | -            | 0.822                                            | 0.815                                                        | **0.869**        | <u>0.847</u>       | 0.825                      |
| TFNS [3]        | -            | 0.331                                            | 0.738                                                        | 0.722            | **0.865**          | <u>0.823</u>               |
| NWGI [4]        | -            | 0.560                                            | 0.588                                                        | **0.674**        | <u>0.593</u>       | 0.578                      |
| Mean            |              | 0.544                                            | 0.735                                                        | 0.760            | **0.789 **         | <u>0.772</u>               |
| Std             |              | 0.180                                            | <u>0.090</u>                                                 | **0.072**        | 0.113              | 0.108                      |
| **Macro F1**    |              |                                                  |                                                              |                  |                    |                            |
| FPB [1]         | -            | 0.487                                            | 0.774                                                        | 0.778            | **0.834**          | <u>0.812</u>               |
| FiQA-SA [2]     | -            | 0.560                                            | 0.665                                                        | **0.796**        | <u>0.767</u>       | 0.705                      |
| TFNS [3]        | -            | 0.340                                            | 0.681                                                        | 0.653            | **0.828**          | <u>0.778</u>               |
| NWGI [4]        | -            | 0.489                                            | 0.579                                                        | **0.646**        | <u>0.599</u>       | 0.572                      |
| Mean            |              | 0.469                                            | 0.675                                                        | <u>0.718</u>     | **0.757**          | 0.717                      |
| Std             |              | 0.080                                            | <u>**0.069**</u>                                             | <u>**0.069**</u> | 0.095              | 0.092                      |
| **Weighted F1** |              |                                                  |                                                              |                  |                    |                            |
| FPB [1]         | 0.511        | 0.381                                            | 0.795                                                        | 0.780            | **0.851**          | <u>0.829</u>               |
| FiQA-SA [2]     | 0.751        | 0.790                                            | 0.806                                                        | **0.868**        | <u>0.853</u>       | 0.827                      |
| TFNS [3]        | -            | 0.189                                            | 0.740                                                        | 0.721            | **0.865**          | <u>0.822</u>               |
| NWGI [4]        | -            | 0.449                                            | 0.578                                                        | **0.710**        | <u>0.587</u>       | 0.567                      |
| Mean            |              | 0.452                                            | 0.730                                                        | <u>0.770</u>     | **0.789**          | 0.761                      |
| Std             |              | 0.217                                            | <u>0.091</u>                                                 | **0.063**        | 0.117              | 0.112                      |

​	**X**: Best score, <u>X</u>: Second best score.

[[1] Financial_Phrasebank (FPB) ](https://huggingface.co/datasets/financial_phrasebank) is a financial news sentiment analysis benchmark, the labels are "positive", "negative" and "neutral". We use the same split as BloombergGPT. BloombergGPT only use 5-shots in the test to show their model's outstanding performance without further finetuning. However, is our task, all data in the 'train' part were used in finetuning, So our results are far better than Bloomberg's.

[[2] FiQA SA](https://huggingface.co/datasets/pauri32/fiqa-2018) consists of 17k sentences from microblog headlines and financial news. These labels were changed to "positive", "negative" and "neutral" according to BloombergGPT's paper. We have tried to use the same split as BloombergGPT's paper. However, the amounts of each label can't match exactly when the seed was set to 42.

[[3] Twitter Financial News Sentiment (TFNS)](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) dataset is an English-language dataset containing an annotated corpus of finance-related tweets. This dataset is used to classify finance-related tweets for their sentiment. The dataset holds 11,932 documents annotated with 3 labels: "Bearish" ("negative"), "Bullish" ("positive"), and "Neutral".

[[4] News With GPT Instruction (NWGI)](https://huggingface.co/datasets/oliverwang15/news_with_gpt_instructions) is a dataset whose labels were generated by ChatGPT. The train set has 16.2k samples and the test set has 4.05k samples. The dataset not only contains 7 classification labels: "strong negative", "moderately negative", "mildly negative", "neutral", "mildly positive", "moderately positive", "strong positive". but it also has the reasons for that result, which might be helpful in the instruction finetuning.

## Ⅲ. Train & Test set

* The training set and testing set are all the split from the four datasets, so this task is actually full-shots instead of few-shots or zero-shots.

| Dataset     | Dataset Training samples | duplication | Total Training samples | part% | Test samples |
| ----------- | ------------------------ | ----------- | ---------------------- | ----- | ------------ |
| FPB [1]     | 3634                     | 6           | 21804                  | 28.4  | 1212         |
| FiQA-SA [2] | 938                      | 21          | 19698                  | 25.7  | 275          |
| TFNS [3]    | 9543                     | 2           | 19086                  | 24.9  | 2388         |
| NWGI [4]    | 16184                    | 1           | 16184                  | 21.0  | 4047         |
| Total       | -                        | -           | 76772                  | 100   | -            |

## Ⅳ. Models settings

* LoRA setting for all models:

  ```python
  peft_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      inference_mode=False,
      r=8,
      lora_alpha=32,
      lora_dropout=0.1,
      target_modules=['query_key_value'],
      bias='none',
  )
  ```

### 1. FinGPT v3

* Training in bf16 with deepspeed with 8*A100 in 7h21min
* Training args:

  ``` python
  training_args = TrainingArguments(
      output_dir='./finetuned_model',
      logging_steps = 100,
      max_steps=10000,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=8,
      learning_rate=1e-5,
      weight_decay=0.01,
      warmup_steps=10000,
      save_steps=5000,
      bf16=True,
      deepspeed=deepspeed_config,
      torch_compile = True,
      load_best_model_at_end = True,
      evaluation_strategy="steps",
      remove_unused_columns=False,
  )
  ```

### 2. FinGPT v3.1

* Training in fp16 with deepspeed with 8*A100 in 2h35min

* Training args:

  ``` python
  training_args = TrainingArguments(
      output_dir='./finetuned_model',
      logging_steps = 100,
      # max_steps=10000,
      num_train_epochs = 2*8,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=8,
      learning_rate=1e-4,
      weight_decay=0.01,
      warmup_steps=1500,
      save_steps=500,
      fp16=True,
      deepspeed=deepspeed_config,
      torch_compile = True,
      load_best_model_at_end = True,
      evaluation_strategy="steps",
      remove_unused_columns=False,
  )
  ```

### 3. FinGPT v3.2 (8bit)

* Training in int8 with 1*RTX3090 in 6h28min

* Training args:

  ``` python
  training_args = TrainingArguments(
      output_dir='./finetuned_model',
      logging_steps = 500,
      # max_steps=10000,
      num_train_epochs = 2,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=8,
      learning_rate=1e-4,
      weight_decay=0.01,
      warmup_steps=1000,
      save_steps=500,
      fp16=True,
      torch_compile = False,
      load_best_model_at_end = True,
      evaluation_strategy="steps",
      remove_unused_columns=False,
  )
  ```

### 4. FinGPT v3.3 (4-bit  QLoRA)

* Training in int4 with 1*RTX3090 in 4h9min

* Training args:

  ```python
  training_args = TrainingArguments(
      output_dir='./finetuned_model',
      logging_steps = 500,
      # max_steps=10000,
      num_train_epochs = 2,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=8,
      learning_rate=1e-4,
      weight_decay=0.01,
      warmup_steps=10000,
      save_steps=5000,
      fp16=True,
      torch_compile = False,
      load_best_model_at_end = True,
      evaluation_strategy="steps",
      remove_unused_columns=False,
  )
  ```

## Ⅴ. How to Train

Coming Soon.

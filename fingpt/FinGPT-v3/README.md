# FinGPT v3 Series

**FinGPT v3 series are LLMs finetuned with LoRA method on the News and Tweets sentiment analysis dataset which achieve best scores on most of the financial sentiment analysis datasets.**

### You can reproduce the results of our experiment by running [benchmarks](./benchmark/benchmarks.ipynb), the detailed tutorial is on the way.

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

| **Weighted F1** | BloombergGPT | ChatGPT | GPT-4 | [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B) | [v3](https://huggingface.co/oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT) | [v3.1](https://huggingface.co/oliverwang15/FinGPT_v31_ChatGLM2_Sentiment_Instruction_LoRA_FT)      | v3.1.1 (8bit) | v3.1.2 (QLoRA) | [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | [v3.2](https://huggingface.co/oliverwang15/FinGPT_v32_Llama2_Sentiment_Instruction_LoRA_FT) | v3.3 (openai) |
| --------------- | :----------: | :----------------------------------------------: | :----------------------------------------------------------: | :--------------: | :----------------: | :------------------------: | :------------------------: | :------------------------: | :------------------------: | :------------------------: | :------------------------: |
| FPB [1] | 0.511 | 0.781 | 0.833 | 0.381 | 0.701 | <ins>**0.855**</ins> | <ins>**0.855**</ins> | 0.777 | 0.390 | *0.850* | **0.878** |
| FiQA-SA [2] | 0.751 | 0.730 | 0.630 | 0.790 | 0.760 | <ins>0.850</ins> | *0.847* | 0.752 | 0.800 | **0.860** | **0.887** |
| TFNS [3] | - | 0.736 | 0.808 | 0.189 | 0.740 | *0.875* | <ins>0.879</ins> | 0.828 | 0.296 | **0.894** | **0.883** |
| NWGI [4] | - | - | - | 0.449 | 0.578 | **0.642** | <ins>0.636</ins> | 0.583 | 0.503 | *0.632* | - |
| Mean | - | - | - | 0.452 | 0.730 | <ins>0.805</ins> | *0.804* | 0.735 | 0.497 | **0.809** | - |
| Std | - | - | - | 0.217 | **0.091** | *0.095* | 0.098 | <ins>0.092</ins> | 0.189 | 0.103 | - |
| **ACC/F1 Micro** |  |  |  |  |  |  |  |  |  |  |  |
| FPB [1]         | -            |  0.781  | 0.834 | 0.464                                            | 0.775                                                        | <ins>**0.856**</ins> | <ins>**0.856**</ins> |      0.797       |                            0.462                             |     *0.851*      | **0.880** |
| FiQA-SA [2]     | -            |  0.662  | 0.545 | 0.822                                            | 0.731                                                        | <ins>*0.836*</ins> |  <ins>*0.836*</ins>  |      0.713       |                            0.822                             |    **0.844**     | **0.865** |
| TFNS [3]        | -            |  0.731  | 0.813 | 0.331                                            | 0.738                                                        | *0.876* | <ins>0.879</ins> |      0.837       |                            0.386                             |    **0.894**     | **0.882** |
| NWGI [4]        | -            | -           | -           | 0.560                                            | 0.588                                                        |      **0.642**       | <ins>0.638</ins> | 0.587 | 0.583 | *0.636* | - |
| Mean            | - | - | - | 0.544                                            | 0.735                                                        |   <ins>0.803</ins>   | *0.802* | 0.733 | 0.563 | **0.806** | - |
| Std             | - | - | - | 0.180                                            | **0.090**                                            | <ins>0.094</ins> |       *0.096*        | *0.096* |                            0.165                             |      0.100       |      -      |
| **Macro F1**    |              |              |              |                                                  |                                                              |                  |                    |                            |                            |                            |                            |
| FPB [1]         | -            | 0.770      | 0.827 | 0.487                                            | 0.745                                                        |      **0.841**       |       *0.837*        |      0.752       |                            0.517                             | <ins>0.840</ins> | **0.873** |
| FiQA-SA [2]     | -            | 0.611       | 0.539 | 0.560                                            | 0.641                                                        |   <ins>0.746</ins>   |       *0.743*        |      0.641       |                            0.610                             |    **0.752**     | **0.781** |
| TFNS [3]        | -            |  0.693  | 0.758 | 0.340                                            | 0.681                                                        |       *0.841*        |   <ins>0.845</ins>   |      0.774       |                            0.401                             |    **0.866**     | **0.858** |
| NWGI [4]        | -            | -           | -           | 0.489                                            | 0.579                                                        |      **0.650**       |   <ins>0.645</ins>   |      0.592       |                            0.539                             |     *0.644*      |     -     |
| Mean            | - | - | - | 0.469                                            | 0.675                                                        |   <ins>0.769</ins>   |       *0.767*        |      0.690       |                            0.517                             |    **0.776**     |    -    |
| Std             | - | - | - | 0.080                                            | **0.069**                                             | 0.079 |        0.081         |     *0.076*      |                       <ins>0.075</ins>                       |      0.087       |      -      |

​	**X**: Best score, <ins>X</ins>: Second best score, *X*: Third best score

​	v3, v3.1, v3.1.1, v3.1.2 is based on ChatGLM2 and v3.2 is based on Llama2

[[1] Financial_Phrasebank (FPB) ](https://huggingface.co/datasets/financial_phrasebank) is a financial news sentiment analysis benchmark, the labels are "positive", "negative" and "neutral". We use the same split as BloombergGPT. BloombergGPT only use 5-shots in the test to show their model's outstanding performance without further finetuning. However, is our task, all data in the 'train' part were used in finetuning, So our results are far better than Bloomberg's.

[[2] FiQA SA](https://huggingface.co/datasets/pauri32/fiqa-2018) consists of 17k sentences from microblog headlines and financial news. These labels were changed to "positive", "negative" and "neutral" according to BloombergGPT's paper. We have tried to use the same split as BloombergGPT's paper. However, the amounts of each label can't match exactly when the seed was set to 42.

[[3] Twitter Financial News Sentiment (TFNS)](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) dataset is an English-language dataset containing an annotated corpus of finance-related tweets. This dataset is used to classify finance-related tweets for their sentiment. The dataset holds 11,932 documents annotated with 3 labels: "Bearish" ("negative"), "Bullish" ("positive"), and "Neutral".

[[4] News With GPT Instruction (NWGI)](https://huggingface.co/datasets/oliverwang15/news_with_gpt_instructions) is a dataset whose labels were generated by ChatGPT. The train set has 16.2k samples and the test set has 4.05k samples. The dataset not only contains 7 classification labels: "strong negative", "moderately negative", "mildly negative", "neutral", "mildly positive", "moderately positive", "strong positive". but it also has the reasons for that result, which might be helpful in the instruction finetuning.

## Ⅲ.  How to Train

* **Notice: The following code is for ChatGLM2 only. if you need to train other models like Llama2, please remember to change the model name in the code:**

  ``` 
  # model_name = "THUDM/chatglm2-6b"
  model_name = "daryl149/llama-2-7b-chat-hf"
  ```

* Prepare for the training data: Run this [notebook](./data/making_data.ipynb)

* Rich Computing Resources (like 8 * A100):

  ``` shell
  cd training_parallel
  sh train.sh
  ```

* Limited Computing Resources:

  * Run this [notebook](./training_8bit/train.ipynb) in 8bit
  * Run this [notebook](./training_int4/train.ipynb) in int4 (QLoRA)

## Ⅳ. Train & Test set

* The training set and testing set are all the split from the four datasets, so this task is actually full-shots instead of few-shots or zero-shots.

| Dataset     | Dataset Training samples | Duplication | Total Training samples | Part% | Test samples |
| ----------- | ------------------------ | ----------- | ---------------------- | ----- | ------------ |
| FPB [1]     | 3634                     | 6           | 21804                  | 28.4  | 1212         |
| FiQA-SA [2] | 938                      | 21          | 19698                  | 25.7  | 275          |
| TFNS [3]    | 9543                     | 2           | 19086                  | 24.9  | 2388         |
| NWGI [4]    | 16184                    | 1           | 16184                  | 21.0  | 4047         |
| Total       | -                        | -           | 76772                  | 100   | -            |

## Ⅴ. Models settings

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

* Training in bf16 with deepspeed with 8*A100 in 7h21min (The training time might be longer than usual due to high CPU usage by other programs)
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

### 3. FinGPT v3.1.1 (8bit)

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

### 4. FinGPT v3.1.2 (4-bit  QLoRA)

* Training in 4bit with 1*RTX3090 in 4h9min

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

### 5. FinGPT v3.2

* Training in bf16 with deepspeed with 8*A100 in 1h03min

* Training args:

  ``` python
  training_args = TrainingArguments(
      output_dir='./finetuned_model'
      logging_steps = 500,
      # max_steps=10000,
      num_train_epochs = 2*4,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=8,
      learning_rate=1e-4,
      weight_decay=0.01,
      warmup_steps=500,
      save_steps=500,
      fp16=True,
      deepspeed=deepspeed_config,
      torch_compile = True,
      load_best_model_at_end = True,
      evaluation_strategy="steps",
      remove_unused_columns=False,
  )
  ```


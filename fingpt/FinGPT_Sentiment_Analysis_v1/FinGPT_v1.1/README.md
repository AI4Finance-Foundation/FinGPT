# FinGPT-V1.1

## Ⅰ. Results & Findings

|             Metrics $^{[1]}$             | Llama2-chat | Ours  | Improvement |
| :-------------------------------------: | :---------: | :---: | :---------: |
| Accuracy / Micro f1 (7 classes $^{[2]}$ ) |    12.61    | 14.59 |   15.7 %    |
|     Weighted F1 (7 classes $^{[2]}$ )     |    4.17     | 11.11 |    166 %    |
|      Macro F1 (7 classse $^{[2]}$ )       |    0.16     | 9.23  |   1000+ %   |
| Accuracy / Micro f1 (3 classes $^{[3]}$ ) |    26.85    | 41.35 |    54 %     |
|     Weighted F1 (3 classes $^{[3]}$ )     |    14.29    | 35.67 |   149.6 %   |
|      Macro F1 (3 classes $^{[3]}$ )       |    12.69    | 22.73 |   79.12 %   |

$^{[1]}$ The groud-true label for the result is generated from the market, please refer to 2.3 section.

$^{[2]}$ 7 classse means the result is one of `Severely Negative`, `Moderately Negative`, `Mildly Negative`, `Neutral`, `Mildly Positive`, `Moderately Positive`, `Severely Positive`. 

$^{[3]}$ For 3 classes, `Severely Negative` and `Moderately Negative` are considered `Negative`; `Mildly Negative`,  `Neutral` and `Mildly Positive` are considered `Neutral`; `Severely Positive`and `Moderately Positive` are considered `Positive`

* The analysis of LLM itself might not align with the market, but we are able to finetune our model to make it align with the market.

## Ⅱ. Data

### 2.1 Data overview (news)

* The data are gathered from online open data sources with exact timestamp. 

* It was split into the training and testing period as follow:

  ``` 
  train_start_date = '2019-01-01'
  train_end_date = '2021-12-31'
  test_start_date = '2022-01-01'
  test_end_date = '2023-08-31'
  ```

### 2.2 Data Aggregation

* To make things better, the best way is to use the new title with the new content. However, it would exceed the max length of 4096, so we get rid of the parts that are too long
* The **News Title** of the news was selected to shorten the total length in order to take more news into consideration.

### 2.3 Label Generation

* The Label was set according to **5-day price change(FDPC)**, it follows the rules below:

  | (-∞, -0.06)       | [-0.06, -0.03)      | [-0.03, -0.01)  | [-0.01, 0.01) | [0.01, 0.03)    | [0.03, 0.06)        | [0.06, +∞)        |
  | ----------------- | ------------------- | --------------- | ------------- | --------------- | ------------------- | ----------------- |
  | Severely Negative | Moderately Negative | Mildly Negative | Neutral       | Mildly Positive | Moderately Positive | Severely Positive |

## Ⅲ. Experiment setting

* Model setting:

  ``` python
  model_name = "daryl149/llama-2-13b-chat-hf"  
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  tokenizer.pad_token_id = tokenizer.eos_token_id
  model =  AutoModelForCausalLM.from_pretrained(
          model_name, 
          trust_remote_code=True, 
          device_map='auto',
      )
  ```

* Training args:

   ``` python
   training_args = TrainingArguments(
           output_dir='./finetuned_model',    # saved model path
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
           # bf16=True,
           torch_compile = False,
           load_best_model_at_end = True,
           evaluation_strategy="steps",
           remove_unused_columns=False,
   )
   ```

* LoRA args:

   ``` python
   target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama']
   lora_config = LoraConfig(
       task_type=TaskType.CAUSAL_LM,
       inference_mode=False,
       r=8,
       lora_alpha=32,
       lora_dropout=0.1,
       target_modules=target_modules,
       bias='none',
   )
   ```

   


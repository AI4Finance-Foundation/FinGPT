# FinGPT's Benchmark

[FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets
](https://arxiv.org/abs/2310.04793)


The datasets we used, and the multi-task financial LLMs models are available at <https://huggingface.co/FinGPT>

---

Before you start, make sure you have the correct versions of the key packages installed.
```
transformers==4.32.0
peft==0.5.0
```

[Weights & Biases](https://wandb.ai/site) is a good tool for tracking model training and inference, you need to register, get a free API, and create a new project.

wandb produces some nice charts like the following:

<img width="440" alt="image" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/04a08b3d-58e3-47aa-8b07-3ec6ff9dfea4">
<img width="440" alt="image" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/f207a64b-622d-4a41-8e0f-1959a2d25450">
<img width="440" alt="image" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/e7699c64-7c3c-4130-94b3-59688631120a">
<img width="440" alt="image" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/65ca7853-3d33-4856-80e5-f03476efcc78">


## Ready-to-use Demo

For users who want ready-to-use financial multi-task language models, please refer to `demo.ipynb`.
Following this notebook, you're able to test Llama2-7B, ChatGLM2-6B, MPT-7B, BLOOM-7B, Falcon-7B, or Qwen-7B with any of the following tasks: 
- Financial Sentiment Analysis
- Headline Classification
- Named Entity Recognition
- Financial Relation Extraction

We suggest users follow the instruction template and task prompts that we used in our training process. Demos are shown in `demo.ipynb`. Due to the limited diversity of the financial tasks and datasets we used, models might not respond correctly to out-of-scope instructions. We'll delve into the generalization ability more in our future works.

## Prepare Data & Base Models

For the base models we used, we recommend pre-downloading them and save to `base_models/`.

Refer to the `parse_model_name()` function in `utils.py` for the huggingface models we used for each LLM. (We use base models rather than any instruction-tuned version or chat version, except for ChatGLM2)

---

For the datasets we used, download our processed instruction tuning data from huggingface. Take FinRED dataset as an example:
```
import datasets

dataset = datasets.load_dataset('FinGPT/fingpt-finred')
# save to local disk space (recommended)
dataset.save_to_disk('data/fingpt-finred')
```
Then `finred` became an available task option for training.

We use different datasets at different phases of our instruction tuning paradigm.
- Task-specific Instruction Tuning: `sentiment-train / finred-re / ner / headline`
- Multi-task Instruction Tuning: `sentiment-train & finred & ner & headline`
- Zero-shot Aimed Instruction Tuning: `finred-cls & ner-cls & headline-cls -> sentiment-cls (test)`

You may download the datasets according to your needs. We also provide processed datasets for ConvFinQA and FinEval, but they are not used in our final work.

### prepare data from scratch
To prepare training data from raw data, you should follow `data/prepate_data.ipynb`. 

We don't include any source data from other open-source financial datasets in our repository. So if you want to do it from scratch, you need to find the corresponding source data and put them in `data/` before you start. 

---

## Instruction Tuning

`train.sh` contains examples of instruction tuning with this repo.
If you don't have training data & base models in your local disk, pass `--from_remote true` in addition.

### Task-specific Instruction Tuning
```
#chatglm2
deepspeed train_lora.py \
--run_name headline-chatglm2-linear \
--base_model chatglm2 \
--dataset headline \
--max_length 512 \
--batch_size 4 \
--learning_rate 1e-4 \
--num_epochs 8
```

Please be aware that "localhost:2" refers to a particular GPU device.

```
#llama2-13b
deepspeed -i "localhost:2" train_lora.py \
--run_name sentiment-llama2-13b-8epoch-16batch \
--base_model llama2-13b-nr \
--dataset sentiment-train \
--max_length 512 \
--batch_size 16 \
--learning_rate 1e-5 \
--num_epochs 8 \
--from_remote True \
>train.log 2>&1 &
```

use 
```
tail -f train.log
```
to check the training log

### Multi-task Instruction Tuning
```
deepspeed train_lora.py \
--run_name MT-falcon-linear \
--base_model falcon \
--dataset sentiment-train,headline,finred*3,ner*15 \
--max_length 512 \
--batch_size 4 \
--learning_rate 1e-4 \
--num_epochs 4
```
### Zero-shot Aimed Instruction Tuning
```
deepspeed train_lora.py \
--run_name GRCLS-sentiment-falcon-linear-small \
--base_model falcon \
--test_dataset sentiment-cls-instruct \
--dataset headline-cls-instruct,finred-cls-instruct*2,ner-cls-instruct*7 \
--max_length 512 \
--batch_size 4 \
--learning_rate 1e-4 \
--num_epochs 1 \
--log_interval 10 \
--warmup_ratio 0 \
--scheduler linear \
--evaluation_strategy steps \
--eval_steps 100 \
--ds_config config_hf.json
```

---

## Evaluation for Financial Tasks

Refer to `Benchmarks/evaluate.sh` for evaluation script on all Financial Tasks.
You can evaluate your trained model on multiple tasks together. For example:
```
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi,headline,ner,re \
--base_model llama2 \
--peft_model ../finetuned_models/MT-llama2-linear_202309241345 \
--batch_size 8 \
--max_length 512
```

```
#llama2-13b sentiment analysis
CUDA_VISIBLE_DEVICES=1 python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--base_model llama2-13b-nr \
--peft_model ../finetuned_models/sentiment-llama2-13b-8epoch-16batch_202310271908  \
--batch_size 8 \
--max_length 512 \
--from_remote True 
```

For Zero-shot Evaluation on Sentiment Analysis, we use multiple prompts and evaluate each of them.
The task indicators are `fiqa_mlt` and `fpb_mlt`.



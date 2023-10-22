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
deepspeed train_lora.py \
--run_name headline-chatglm2-linear \
--base_model chatglm2 \
--dataset headline \
--max_length 512 \
--batch_size 4 \
--learning_rate 1e-4 \
--num_epochs 8
```
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

For Zero-shot Evaluation on Sentiment Analysis, we use multiple prompts and evaluate each of them.
The task indicators are `fiqa_mlt` and `fpb_mlt`.



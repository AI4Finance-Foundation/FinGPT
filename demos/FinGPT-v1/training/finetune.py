from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os

model_name = "/root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/658202d88ac4bb782b99e99ac3adff58b4d0b813"
# model_name = "THUDM/chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="/root/FinGPT-ChatGLM-Fineturning/data/title/dataset_title_train_and_valid")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        # seq_len = feature["seq_len"]
        seq_len = len(ids)
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def prediction_step(self, model: nn.Module, inputs, prediction_loss_only: bool, ignore_keys = None):
        with torch.no_grad():
            res = model(
                input_ids=inputs["input_ids"],
                labels=inputs["labels"],
            ).loss
        return (res, None, None)

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def main():
    writer = SummaryWriter()
    TrainingArguments.torch_compile = True
    TrainingArguments.load_best_model_at_end = True
    TrainingArguments.eval_steps = 50
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # init model
    model = AutoModel.from_pretrained(
        model_name, load_in_8bit=True, trust_remote_code=True, device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value"],
    )
    model = get_peft_model(model, peft_config)

    # load dataset
    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    # dataset = datasets.Dataset.from_dict(dataset[::10])
    train_val = dataset.train_test_split( test_size = 0.1, shuffle=True, seed=42 )
    train_data = train_val["train"].shuffle()
    val_data = train_val["test"].shuffle()
    print(f"\n{len(dataset)=}\n")

    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset = train_data,
        eval_dataset = val_data,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    trainer.train()
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

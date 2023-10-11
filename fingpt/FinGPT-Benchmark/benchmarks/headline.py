from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import datasets
import torch
from torch.utils.data import DataLoader
from functools import partial


import sys
sys.path.append('../')
from utils import *
    
    
    
def binary2multi(dataset):
    pred, label = [], []
    tmp_pred, tmp_label = [], []
    for i, row in dataset.iterrows():
        tmp_pred.append(row['pred'])
        tmp_label.append(row['label'])
        if (i + 1) % 9 == 0:
            pred.append(tmp_pred)
            label.append(tmp_label)
            tmp_pred, tmp_label = [], []
    return pred, label


def map_output(feature):
    pred = 1 if 'yes' in feature['out_text'].lower() else 0
    label = 1 if 'yes' in feature['output'].lower() else 0
    return {'label': label, 'pred': pred}


def test_headline(args, model, tokenizer):
    
    # dataset = load_from_disk('../data/fingpt-headline')['test']#.select(range(300))
    dataset = load_from_disk('../data/fingpt-headline-instruct')['test']#.select(range(300))
    dataset = dataset.map(partial(test_mapping, args), load_from_cache_file=False)
    
    def collate_fn(batch):
        inputs = tokenizer(
            [f["prompt"] for f in batch], return_tensors='pt',
            padding=True, max_length=args.max_length,
            return_token_type_ids=False
        )
        return inputs
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    out_text_list = []
    log_interval = len(dataloader) // 5

    for idx, inputs in enumerate(tqdm(dataloader)):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        res = model.generate(**inputs, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        tqdm.write(f'{idx}: {res_sentences[0]}')
        if (idx + 1) % log_interval == 0:
            tqdm.write(f'{idx}: {res_sentences[0]}')
        out_text = [o.split("Answer: ")[1] for o in res_sentences]
        out_text_list += out_text
        torch.cuda.empty_cache()
    
    dataset = dataset.add_column("out_text", out_text_list)
    dataset = dataset.map(map_output, load_from_cache_file=False)    
    dataset = dataset.to_pandas()
    
    print(dataset)
    dataset.to_csv('tmp.csv')
        
    # binary
    acc = accuracy_score(dataset["label"], dataset["pred"])
    f1 = f1_score(dataset["label"], dataset["pred"], average="binary")
    
    # multi-class
    pred, label = binary2multi(dataset)

    print(f"\n|| Acc: {acc} || F1 binary: {f1} ||\n")
    print(classification_report(label, pred, digits=4, target_names=['price or not', 'price up', 'price stable',
                                                                     'price down', 'price past', 'price future',
                                                                     'event past', 'event future', 'asset comp']))

    return dataset
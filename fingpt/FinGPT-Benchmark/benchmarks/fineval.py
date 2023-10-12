from seqeval.metrics import accuracy_score
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import datasets
import torch
from torch.utils.data import DataLoader
from functools import partial
import re
import sys
import numpy as np
sys.path.append('../')
from utils import *
    

def cvt_text_to_pred(text):
    
    pred_match = re.search(r'[ABCD]', text)
    if pred_match is not None:
        pred = pred_match.group()
        pred = ["A", "B", "C", "D"].index(pred)
    else:
        pred = -1
    return pred


def map_output(feature):

    label = cvt_text_to_pred(feature['output'])
    pred = cvt_text_to_pred(feature['out_text'])
    
    return {'label': label, 'pred': pred}


def test_fineval(args, model, tokenizer):

    dataset = load_from_disk('../data/fingpt-fineval')['test']#.select(range(30))
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
    
    print('Accuracy:', accuracy_score(dataset['label'], dataset['pred']))

    return dataset
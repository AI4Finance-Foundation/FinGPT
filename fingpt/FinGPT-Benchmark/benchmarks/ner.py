from seqeval.metrics import classification_report
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
    

ent_dict = {
    'PER': 'person',
    'ORG': 'organization',
    'LOC': 'location',
}
ent_dict_rev = {v: k for k, v in ent_dict.items()}


def cvt_text_to_pred(tokens, text):
    
    preds = ['O' for _ in range(len(tokens))]
    for pred_txt in text.lower().strip('.').split(','):
    
        pred_match = re.match(r'^(.*) is an? (.*)$', pred_txt)
        if pred_match is not None:
            entity, entity_type = pred_match.group(1).strip(), pred_match.group(2).strip()
            entity_pred = ent_dict_rev.get(entity_type, 'O')
            entity_tokens = entity.split()

            n = len(entity_tokens)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i+n] == entity_tokens and preds[i:i+n] == ['O'] * n:
                    preds[i:i+n] = ['B-' + entity_pred] + ['I-' + entity_pred] * (n-1)
                    break
        else:
            print(pred_txt)
            
    return preds


def map_output(feature):

    tokens = feature['input'].lower().split()
    label = cvt_text_to_pred(tokens, feature['output'])
    pred = cvt_text_to_pred(tokens, feature['out_text'])
    
    return {'label': label, 'pred': pred}


def test_ner(args, model, tokenizer):

    dataset = load_from_disk('../data/fingpt-ner')['test']#.select(range(30))
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
    
    label = [d.tolist() for d in dataset['label']]
    pred = [d.tolist() for d in dataset['pred']]
    
    print(classification_report(label, pred, digits=4))

    return dataset
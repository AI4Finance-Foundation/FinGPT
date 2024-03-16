import datasets
from pathlib import Path
import argparse

DATASETS = [
    # source, destination
    (('pauri32/fiqa-2018', None), 'fiqa-2018'),
    (('FinGPT/fingpt-finred', None), 'fingpt-finred'),
    (('zeroshot/twitter-financial-news-sentiment', None), 'twitter-financial-news-sentiment'),
    (('oliverwang15/news_with_gpt_instructions', None), 'news_with_gpt_instructions'),
    (('financial_phrasebank', 'sentences_50agree'), 'financial_phrasebank-sentences_50agree'),
    (('FinGPT/fingpt-fiqa_qa', None), 'fingpt-fiqa_qa'),
    (('FinGPT/fingpt-headline-cls', None), 'fingpt-headline-cls'),
    (('FinGPT/fingpt-finred', None), 'fingpt-finred'),
    (('FinGPT/fingpt-convfinqa', None), 'fingpt-convfinqa'),
    (('FinGPT/fingpt-finred-cls', None), 'fingpt-finred-cls'),
    (('FinGPT/fingpt-ner', None), 'fingpt-ner'),
    (('FinGPT/fingpt-headline', None), 'fingpt-headline-instruct'),
    (('FinGPT/fingpt-finred-re', None), 'fingpt-finred-re'),
    (('FinGPT/fingpt-ner-cls', None), 'fingpt-ner-cls'),
    (('FinGPT/fingpt-fineval', None), 'fingpt-fineval'),
    (('FinGPT/fingpt-sentiment-cls', None), 'fingpt-sentiment-cls'),
]

def download(no_cache: bool = False):
    """Downloads all datasets to where the FinGPT library is located."""
    data_dir = Path(__file__).parent
    
    for src, dest in DATASETS:
        if Path(data_dir / dest).is_dir() and not no_cache:
            print(f"Dataset found at {data_dir / dest}, skipping")
            continue
        dataset = datasets.load_dataset(*src)
        dataset.save_to_disk(data_dir / dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cache", default=False, required=False, type=str, help="Redownloads all datasets if set to True")
    
    args = parser.parse_args()
    download(no_cache=args.no_cache)

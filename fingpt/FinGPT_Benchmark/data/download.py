import datasets
from pathlib import Path

def download():
    """Downloads all datasets to where the FinGPT library is located."""
    data_dir = Path(__file__).parent
    
    dataset = datasets.load_dataset('pauri32/fiqa-2018')
    dataset.save_to_disk(data_dir / 'fiqa-2018')
    
    dataset = datasets.load_dataset('FinGPT/fingpt-finred')
    dataset.save_to_disk(data_dir / 'fingpt-finred')
    
    dataset = datasets.load_dataset('zeroshot/twitter-financial-news-sentiment')
    dataset.save_to_disk(data_dir / 'twitter-financial-news-sentiment')
    
    dataset = datasets.load_dataset('oliverwang15/news_with_gpt_instructions')
    dataset.save_to_disk(data_dir / 'news_with_gpt_instructions')

    dataset = datasets.load_dataset("financial_phrasebank", "sentences_50agree")
    dataset.save_to_disk(data_dir / 'financial_phrasebank-sentences_50agree')

    dataset = datasets.load_dataset('FinGPT/fingpt-fiqa_qa')
    dataset.save_to_disk(data_dir / 'fingpt-fiqa_qa')

    dataset = datasets.load_dataset('FinGPT/fingpt-headline-cls')
    dataset.save_to_disk(data_dir / 'fingpt-headline-cls')

    dataset = datasets.load_dataset('FinGPT/fingpt-finred')
    dataset.save_to_disk(data_dir / 'fingpt-finred')

    dataset = datasets.load_dataset('FinGPT/fingpt-convfinqa')
    dataset.save_to_disk(data_dir / 'fingpt-convfinqa')

    dataset = datasets.load_dataset('FinGPT/fingpt-finred-cls')
    dataset.save_to_disk(data_dir / 'fingpt-finred-cls')

    dataset = datasets.load_dataset('FinGPT/fingpt-ner')
    dataset.save_to_disk(data_dir / 'fingpt-ner')

    dataset = datasets.load_dataset('FinGPT/fingpt-headline')
    dataset.save_to_disk(data_dir / 'fingpt-headline-instruct')

    dataset = datasets.load_dataset('FinGPT/fingpt-finred-re')
    dataset.save_to_disk(data_dir / 'fingpt-finred-re')

    dataset = datasets.load_dataset('FinGPT/fingpt-ner-cls')
    dataset.save_to_disk(data_dir / 'fingpt-ner-cls')

    dataset = datasets.load_dataset('FinGPT/fingpt-fineval')
    dataset.save_to_disk(data_dir / 'fingpt-fineval')

    dataset = datasets.load_dataset('FinGPT/fingpt-sentiment-cls')
    dataset.save_to_disk(data_dir / 'fingpt-sentiment-cls')

if __name__ == "__main__":
    download()

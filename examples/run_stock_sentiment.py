"""
Minimal example: download CNBC news for TSLA and show the first rows.
This helps first-time users verify their environment quickly.
"""

from finnlp.data_sources.news.cnbc import CNBC

def main() -> None:
    loader = CNBC()
    # You may change TSLA to any ticker string accepted by CNBC loader
    loader.download_news(stock="TSLA")
    # Print only a few rows to keep output small
    print(loader.data.head())

if __name__ == "__main__":
    main()

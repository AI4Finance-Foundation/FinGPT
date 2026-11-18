from finnlp.data_sources.news.cnbc import CNBC

if __name__ == "__main__":
    loader = CNBC()
    loader.download_news(stock="AAPL")
    print(loader.data.head())

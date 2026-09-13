# FinGPT-V1.0

## Fine-tuning data validation

The v1 fine-tuning script expects a processed Hugging Face dataset created by
`making_dataset/tokenize_dataset_rows.py`, not a raw JSONL directory. The
saved dataset must contain `input_ids` and `seq_len` columns and at least two
examples so the script can create its 90/10 train/validation split.

Before loading the model, `training/finetune.py` checks the dataset path,
required columns, empty token sequences, and invalid sequence lengths. Errors
identify the failed condition and point back to the preprocessing step.

For poor evaluation results, first compare the number of usable examples and
the label distribution in the source JSONL, then verify that the preprocessing
tokenizer and prompt format match the inference code. Only after those checks
should you tune learning rate, batch size, or number of epochs.

### Ⅰ. Data Preparations
#### 1. Download Titles [code](./data_preparations/download_titles.py)
* In this file, we downloaded the financial news titles and URLs from [eastmoney(东方财富)](https://www.eastmoney.com/)
* **Important**: This script uses `Eastmoney_Streaming` from the [FinNLP library](https://github.com/AI4Finance-Foundation/FinNLP)
* You need to clone the FinNLP repository and ensure it's available at `../../FinNLP` relative to this script
* The script expects FinNLP to be in the parent directory structure or you need to adjust the import path

#### 2. Download Content [code](./data_preparations/download_contents.py)
* In this file, we downloaded the financial news Contents from [eastmoney(东方财富)](https://www.eastmoney.com/)

**Troubleshooting Eastmoney Download Issues:**

If you encounter issues downloading eastmoney content (no data downloaded, no errors reported):

1. **Proxy Configuration**: The download scripts use proxy services (kuaidaili) to avoid rate limiting
   - Ensure your proxy credentials (YOUR_KUAIDAILI_TUNNEL, YOUR_KUAIDAILI_USERNAME, YOUR_KUAIDAILI_PASSWARD) are correctly set
   - Check if your proxy service subscription is active
   - Verify proxy connection settings

2. **Rate Limiting**: Eastmoney may have rate limits or anti-scraping measures
   - Reduce the number of concurrent downloads (adjust `processes` in multiprocessing Pool)
   - Add delays between requests
   - Use different proxy servers or rotate IPs

3. **Network Issues**: 
   - Check your internet connection
   - Verify DNS settings
   - Try accessing eastmoney.com directly in a browser

4. **Script Configuration**:
   - Ensure the `max_retry` parameter is set appropriately (default is 5)
   - Check that the stock list (hs_300.csv) contains valid security codes
   - Verify file paths are correct and writable

5. **Alternative Data Sources**: If eastmoney downloads consistently fail:
   - Consider using other financial data sources
   - Use pre-processed datasets if available
   - Check if FinNLP library has alternative data source implementations

#### 3. Add labels [code](./data_preparations/add_labels.py)
* In this file, we add the label for news titles and contents.
* The labels are determined by the change pct between the stock price of today and 5-days later
    * change pct >= 0.06 : `very positive` 
    * 0.02 <= change pct <= 0.06 : `positive`
    * -0.02 <= change pct <= 0.02 : `neutral`
    * -0.06 <= change pct <= -0.02 : `negative` 
    * change pct <= -0.06 : `very negative` 

### Ⅱ. Making Dataset

1. Make dataset_by_date [code](https://github.com/AI4Finance-Foundation/FinGPT/blob/master/fingpt/FinGPT-v1/making_dataset/make_dataset_by_date.ipynb)
   * You may run this notebook to generate the dataset file in alpaca format

2. Please run the following two files respectively to generate the dataset in hugging face dataset format.

   * [change_jsonl_train_and_valid.sh](./making_dataset/change_jsonl_train_and_valid.sh)

   * [make_dataset_train_and_valid.sh](./making_dataset/make_dataset_train_and_valid.sh)

### Ⅲ. Training (Finetuning)
* Please run the following codes
    ``` shell
    cd training
    sh finetune.sh
    ```

#### Expected Training Times
Training time varies significantly based on:
- **Dataset size**: Number of training examples
- **Hardware**: GPU type and memory
- **Model size**: Base model parameter count
- **Configuration**: Batch size, learning rate, epochs

**Typical training times for reference:**
- Small dataset (2 stocks, ~70M examples) on Tesla V100 32GB: A few minutes per epoch
- Full dataset on RTX 3090: Several hours to overnight
- Full dataset on A100: 4-6 hours depending on configuration

**Note**: Very short training times (a few minutes) may indicate:
1. Very small dataset size
2. Insufficient training epochs
3. Hardware not being fully utilized

If training completes unusually quickly, verify:
- Dataset size is appropriate for your use case
- Number of epochs is sufficient (default is 1 in finetune.sh)
- GPU utilization is high during training

### Ⅳ. Inferencing 
* Please refer to [infer.ipynb](./inferencing/infer.ipynb)

### Special thanks to [ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning) for introductions on how to finetune ChatGLM by using huggingface.

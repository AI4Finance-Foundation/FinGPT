# FinGPT FAQ (Getting Started)

### 1) I installed packages but import fails with `ModuleNotFoundError`
Ensure you installed dependencies in the same Python environment you are running.
Confirm with:
python -c "import sys; print(sys.executable)"

If needed, reinstall:
        >> pip install -r requirements.txt

### 2) Which Python version should I use
Prefer a recent Python 3 version supported by the project. If installation fails, try Python 3.9 or later.

### 3) How do I quickly verify my environment
Run the example:
        >> python examples/run_stock_sentiment.py

You should see a small table of CNBC news for TSLA.

### 4) I do not see my contributions under my GitHub profile
Make sure your commit email matches the email registered on your GitHub account. You can use the GitHub no reply email as well.

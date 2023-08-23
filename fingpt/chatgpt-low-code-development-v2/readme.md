# ChatGPT Low Code Development V2: Finding New Factors

In addition to factor development, the quest for identifying effective factors is also a challenging journey. Our demo can expedite this process through the use of tailored prompts. 

## Round 1
### init prompt: 

Now, you are a senior professional stock factor researcher in top financial companies.You will help me to build some factors in Python today.

### ChatGPT's response (More in [this file](./main.ipynb)):

Of course! I'd be happy to help you build some factors in Python. Please let me know what specific factors you would like to build or if you have any specific requirements in mind.

## Round 2
### prompt 1:  
Let's look at the data, it's stored in the CSV format and has the following columns: open, high, low, close, volume, and amount.

### ChatGPT's response (More in [this file](./main.ipynb)):
Great! To begin, we'll start by loading the data from the CSV file into a pandas DataFrame. Assuming the file is named "data.csv" and is located in the current working directory, you can use the following code:

```python
import pandas as pd

# Load data from CSV into a DataFrame
data = pd.read_csv('data.csv')

# Display a preview of the data
print(data.head())
```

...

## Round 3
### prompt 2: 
OK. First, please show me a function to calculate the Fundamental Momentum Factor. Let's use FMF to indicate Fundamental Momentum Factor.

### ChatGPT's response (More in [this file](./main.ipynb)):
The Fundamental Momentum Factor (FMF) is a measure that quantifies the change in fundamental metrics over a specific period. To calculate the FMF, we need to select a fundamental metric (e.g., "amount") and a time period (e.g., 1 year). Here's how you can calculate the FMF in Python:

```python
import pandas as pd

def calculate_fmf(data, metric, period):
    # Convert the date column to datetime format
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort the data by date in ascending order
    data = data.sort_values('date')
    
    # Calculate the FMF
    data['fmf'] = data[metric].pct_change(periods=period)
    
    return data

# Load data from CSV into a DataFrame
data = pd.read_csv('data.csv')

# Calculate FMF using the 'amount' metric over a 1-year period
fmf_data = calculate_fmf(data, 'amount', 252)

# Display a preview of the FMF data
print(fmf_data.head())
```

...

## Round 4

### prompt 3:  

Next, please show me a function to calculate the Price Momentum Factor. Let's use PMF to indicate Price Momentum Factor.

### ChatGPT's response (More in [this file](./main.ipynb)):
The Price Momentum Factor (PMF) is a measure that quantifies the price momentum of a stock. It is typically calculated as the percentage change in the closing price over a specific period. Here's how you can calculate the PMF in Python:

```python
import pandas as pd

def calculate_pmf(data, period):
    # Convert the date column to datetime format
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort the data by date in ascending order
    data = data.sort_values('date')
    
    # Calculate the PMF
    data['pmf'] = data['close'].pct_change(periods=period)
    
    return data

# Load data from CSV into a DataFrame
data = pd.read_csv('data.csv')

# Calculate PMF over a 1-year period
pmf_data = calculate_pmf(data, 252)

# Display a preview of the PMF data
print(pmf_data.head())
```

...

## Round 5

### prompt 4:  

Next, please show me a function to calculate the Price Momentum Factor. Let's use PMF to indicate Price Momentum Factor.

### ChatGPT's response (More in [this file](./main.ipynb)):

Certainly! Let's combine the Fundamental Momentum Factor (FMF) and Price Momentum Factor (PMF) functions to create a more complex stock factor. Here's an example of how we can do it step by step:

1. Calculate FMF:
   ```python
   fmf_data = calculate_fmf(data, 'amount', 252)
   ```

2. Calculate PMF:
   ```python
   pmf_data = calculate_pmf(data, 252)
   ```

3. Merge FMF and PMF data on the 'date' column:
   ```python
   merged_data = pd.merge(fmf_data, pmf_data[['date', 'pmf']], on='date', how='inner')
   ```

4. Create a new factor by multiplying FMF and PMF:
   ```python
   merged_data['complex_factor'] = merged_data['fmf'] * merged_data['pmf']
   ```

5. Optionally, you may want to normalize the complex factor to a certain range or scale for easier interpretation. For instance, you can use the Min-Max scaling method:
   ```python
   merged_data['complex_factor_scaled'] = (merged_data['complex_factor'] - merged_data['complex_factor'].min()) / (merged_data['complex_factor'].max() - merged_data['complex_factor'].min())
   ```

...

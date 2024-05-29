# TrendBasedTradingAI

TrendBasedTradingAI is an AI-driven project designed to optimize stock market trading through the use of K-means clustering. This model analyzes stock market trends and makes trading decisions based on the identified trends. By buying shares during upward trends and selling during downward trends, the model aims to maximize returns and minimize risks.

## 1) Features

**1.1) Stock Price Data Collection**: Utilizes Yahoo Finance to download historical stock price data.
**1.2) Technical Indicator Calculation**: Calculates the Relative Strength Index (RSI) and 50-day Exponential Moving Average (EMA) to assist in identifying market trends.
**1.3) K-means Clustering**: Implements K-means clustering to categorize market behavior into distinct trends.
**1.4) Trend Labeling**: Uses linear regression to label each cluster as an upward or downward trend.
**1.5) Test Data Evaluation**: Applies the trained model to test data, categorizing each window into an upward or downward trend.
**1.6) Trading Strategy Implementation**: Simulates a trading strategy based on identified trends, including transaction cost considerations.
**1.7) Performance Metrics**: Evaluates the strategy's performance using metrics such as annual returns and the Sharpe Ratio.

## 2) Usage

The model processes stock market data, calculates essential indicators, and applies K-means clustering to identify trends. It labels these trends and simulates a trading strategy based on the labeled trends, considering transaction costs. The strategy's performance is evaluated using the Sharpe Ratio, providing insights into the risk-adjusted returns of the trading model.

## 3) Key Results

**3.1) Cluster Trend Labeling**: Determines the trend for each cluster using linear regression.
**3.2) Trading Simulation**: Simulates trading actions based on trend predictions.
**3.3) Performance Evaluation**: Calculates returns for each year and evaluates the Sharpe Ratio to assess the strategy's risk-adjusted performance.

This project demonstrates the potential of combining AI techniques with financial data analysis to create an intelligent trading strategy. It provides a foundation for further exploration and refinement in the field of AI-driven stock market trading.

## 4) How to Run the Project

Follow these steps to run the project:

**4.1) Clone the Repository**
Clone the repository to your local machine using the following command:
```bash 
git clone https://github.com/MFaizan18/TrendBasedTradingAI.git
```
**4.2) Navigate to the Project Directory**
```bash 
cd TrendBasedTradingAI
```
**4.3) Install the Required Packages**
```bash 
pip install -r requirements.txt
```
**4.4) Run the Script**
```bash
python TrendBasedTradingAI.py
```
## 5) Model Performance Overview

**5.1) Data Acquisition**

We use the `yfinance` library in Python to download historical stock price data from Yahoo Finance. We evaluate our model on five major indices:

1. ^NSEI - Nifty 50 index on the National Stock Exchange of India
2. ^SPX - S&P 500 index in the USA
3. ^HSI - Hang Seng index in Hong Kong
4. ^FTSE - FTSE 100 index in London
5. ^N225 - Nikkei 225 index in Japan

For the purpose of this explanation, we will use the Nifty 50 index (^NSEI) as an example. The data spans from January 1, 2010, to May 25, 2024, providing us with over a decades of daily stock price data.

```python
import yfinance as yf
```
**5.2) Download stock price data**
data = yf.download("^NSEI", start="2010-01-01", end="2024-05-25", interval="1d")

Here's a glimpse of the data we're working with. The first 10 rows of the data are as follows:
```
| Date                | Open         | High         | Low          | Close        | Adj Close   | Volume |
|---------------------|--------------|--------------|--------------|--------------|-------------|--------|
| 2007-09-17 00:00:00 | 4518.450195  | 4549.049805  | 4482.850098  | 4494.649902  | 4494.649902 | 0      |
| 2007-09-18 00:00:00 | 4494.100098  | 4551.799805  | 4481.549805  | 4546.200195  | 4546.200195 | 0      |
| 2007-09-19 00:00:00 | 4550.25      | 4739         | 4550.25      | 4732.350098  | 4732.350098 | 0      |
| 2007-09-20 00:00:00 | 4734.850098  | 4760.850098  | 4721.149902  | 4747.549805  | 4747.549805 | 0      |
| 2007-09-21 00:00:00 | 4752.950195  | 4855.700195  | 4733.700195  | 4837.549805  | 4837.549805 | 0      |
| 2007-09-24 00:00:00 | 4837.149902  | 4941.149902  | 4837.149902  | 4932.200195  | 4932.200195 | 0      |
| 2007-09-25 00:00:00 | 4939.100098  | 4953.899902  | 4878.149902  | 4938.850098  | 4938.850098 | 0      |
| 2007-09-26 00:00:00 | 4937.600098  | 4980.850098  | 4930.350098  | 4940.5       | 4940.5      | 0      |
| 2007-09-27 00:00:00 | 4942.700195  | 5016.399902  | 4942.700195  | 5000.549805  | 5000.549805 | 0      |
| 2007-09-28 00:00:00 | 5000.25      | 5069.899902  | 4991.149902  | 5063.850098  | 5063.850098 | 0      |

And the last 10 rows of the data are as follows:

| Date                | Open         | High         | Low          | Close        | Adj Close   | Volume |
|---------------------|--------------|--------------|--------------|--------------|-------------|--------|
| 2024-05-09 00:00:00 | 22224.80078  | 22307.75     | 21932.40039  | 21957.5      | 21957.5     | 331300 |
| 2024-05-10 00:00:00 | 21990.94922  | 22131.30078  | 21950.30078  | 22055.19922  | 22055.19922 | 265800 |
| 2024-05-13 00:00:00 | 22027.94922  | 22131.65039  | 21821.05078  | 22104.05078  | 22104.05078 | 278200 |
| 2024-05-14 00:00:00 | 22112.90039  | 22270.05078  | 22081.25     | 22217.84961  | 22217.84961 | 230200 |
| 2024-05-15 00:00:00 | 22255.59961  | 22297.55078  | 22151.75     | 22200.55078  | 22200.55078 | 231900 |
| 2024-05-16 00:00:00 | 22319.19922  | 22432.25     | 22054.55078  | 22403.84961  | 22403.84961 | 368900 |
| 2024-05-17 00:00:00 | 22415.25     | 22502.15039  | 22345.65039  | 22466.09961  | 22466.09961 | 242700 |
| 2024-05-21 00:00:00 | 22404.55078  | 22591.09961  | 22404.55078  | 22529.05078  | 22529.05078 | 347600 |
| 2024-05-22 00:00:00 | 22576.59961  | 22629.5      | 22483.15039  | 22597.80078  | 22597.80078 | 290300 |
| 2024-05-23 00:00:00 | 22614.09961  | 22993.59961  | 22577.44922  | 22967.65039  | 22967.65039 | 369800 |
| 2024-05-24 00:00:00 | 22930.75     | 23026.40039  | 22908        | 22957.09961  | 22957.09961 | 261900 |
```
## 6) Feature Engineering

In this section, we perform feature engineering on our dataset. We calculate and add new features that might be useful for our model.

**6.1) Calculate RSI:** The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. It is used to identify overbought or oversold conditions in a market. We're using the RSIIndicator from the ta library to calculate the RSI with a window of 14 days (which is a common choice) based on the 'Adj Close' prices. The result is then added as a new column 'RSI' to our DataFrame.
```python
# Calculate RSI
rsi_indicator = RSIIndicator(close=data['Adj Close'], window=14)
data['RSI'] = rsi_indicator.rsi()
```
**6.2) Calculate 50-day EMA:** The Exponential Moving Average (EMA) is a type of moving average that gives more weight to recent prices, which can make it more responsive to new information. We're calculating the 50-day EMA based on the 'Adj Close' prices and adding it as a new column '50_EMA' to our DataFrame.
```python
# Calculate 50-day EMA
ema_indicator = EMAIndicator(close=data['Adj Close'], window=50)
data['50_EMA'] = ema_indicator.ema_indicator()
data.dropna(inplace=True)
```
**6.3) Drop NaN values:** Since the EMA requires a certain amount of data to start calculating, the first few rows of our '50_EMA' column will be NaN. We drop these rows with data.dropna(inplace=True).
```python
# Drop the 'Close' column
data = data.drop(columns=['Close'])
```
**6.4) Drop the 'Close' column:** We drop the 'Close' column as we have the 'Adj Close' column which is a more accurate reflection of the stock's value, as it accounts for dividends and stock splits.
```python
# Drop the 'Close' column
data = data.drop(columns=['Close'])
```
**6.7) Store dates in a separate DataFrame:** We store the dates in a separate DataFrame for future use, as we're going to reset the index of our main DataFrame in the next step.
```python
# Store dates in a separate DataFrame
dates = data.index.to_frame(index=False)
```
**6.8) Reset index:** We reset the index of our DataFrame. This is done because we want our index to be a simple numerical index, which can be useful for certain operations or algorithms.
```python
# Reset index
data.reset_index(drop=True, inplace=True)
```
This completes our feature engineering process, and our data is now ready for the next steps of our analysis or modeling.














# TrendBasedTradingAI

TrendBasedTradingAI is an AI-driven project designed to optimize stock market trading through the use of K-means clustering. This model analyzes stock market trends and makes trading decisions based on the identified trends. By buying shares during upward trends and selling during downward trends, the model aims to maximize returns and minimize risks.

## Features

1. **Stock Price Data Collection**: Utilizes Yahoo Finance to download historical stock price data.
2. **Technical Indicator Calculation**: Calculates the Relative Strength Index (RSI) and 50-day Exponential Moving Average (EMA) to assist in identifying market trends.
3. **K-means Clustering**: Implements K-means clustering to categorize market behavior into distinct trends.
4. **Trend Labeling**: Uses linear regression to label each cluster as an upward or downward trend.
5. **Test Data Evaluation**: Applies the trained model to test data, categorizing each window into an upward or downward trend.
6. **Trading Strategy Implementation**: Simulates a trading strategy based on identified trends, including transaction cost considerations.
7. **Performance Metrics**: Evaluates the strategy's performance using metrics such as annual returns and the Sharpe Ratio.

## Usage

The model processes stock market data, calculates essential indicators, and applies K-means clustering to identify trends. It labels these trends and simulates a trading strategy based on the labeled trends, considering transaction costs. The strategy's performance is evaluated using the Sharpe Ratio, providing insights into the risk-adjusted returns of the trading model.

## Key Results

- **Cluster Trend Labeling**: Determines the trend for each cluster using linear regression.
- **Trading Simulation**: Simulates trading actions based on trend predictions.
- **Performance Evaluation**: Calculates returns for each year and evaluates the Sharpe Ratio to assess the strategy's risk-adjusted performance.

This project demonstrates the potential of combining AI techniques with financial data analysis to create an intelligent trading strategy. It provides a foundation for further exploration and refinement in the field of AI-driven stock market trading.

## How to Run the Project

Follow these steps to run the project:

1. **Clone the Repository**
   Clone the repository to your local machine using the following command:
   ```bash
   git clone https://github.com/MFaizan18/TrendBasedTradingAI.git

2. **Navigate to the Project Directory**
   ```bash
   cd TrendBasedTradingAI

4. **Install the Required Packages**
   ```bash
   pip install -r requirements.txt

6. **Run the Script**
   ```bash
   python TrendBasedTradingAI.py



## Model Performance Overview

## Data Preparation

We use the `yfinance` library in Python to download historical stock price data from Yahoo Finance. We evaluate our model on five major indices:

1. ^NSEI - Nifty 50 index on the National Stock Exchange of India
2. ^SPX - S&P 500 index in the USA
3. ^HSI - Hang Seng index in Hong Kong
4. ^FTSE - FTSE 100 index in London
5. ^N225 - Nikkei 225 index in Japan

For the purpose of this explanation, we will use the Nifty 50 index (^NSEI) as an example. The data spans from January 1, 2010, to May 25, 2024, providing us with over a decades of daily stock price data.

```python
import yfinance as yf

# Download stock price data
data = yf.download("^NSEI", start="2010-01-01", end="2024-05-25", interval="1d")




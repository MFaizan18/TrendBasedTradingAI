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
python main.py
```
## 5) Model Performance Overview

**5.1) Let's start by importimg the necessary libraries**

```python
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime
import math 
```

**5.2) Data Acquisition**

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
**5.3) Download stock price data**

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
**6.3) Drop NaN values:** Since the EMA requires a certain amount of data to start calculating, the first few rows of our '50_EMA' column will be NaN. We drop these rows with the ```dropna``` command. This command removes all rows with at least one NaN value in the `data` DataFrame and the changes are made directly to `data`.

```python
# Drop NaN values
data.dropna(inplace=True)
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
**6.7) Selected Features for Clustering**
For clustering, we will be using the following features: ```Open```, ```High```, ```Low```, ```Adj Close```, ```Volume```, ```RSI```, and ```50_EMA```. These features have been chosen based on their relevance and potential to provide valuable insights into stock price movements.

This completes our feature engineering process, and our data is now ready for the next steps of our analysis or modeling.

## 7) Data Segmentation and Normalization

**7.1) Divide Data into Training and Test Sets:** The data is divided into training and test sets based on the year. The training set includes data up to and including the year 2020, and the test set includes data from 2021 onwards.

```python
training_data = data[dates['Date'].dt.year <= 2020].copy()
test_data = data[dates['Date'].dt.year > 2020].copy()
```
**7.2) Define Window Lengths:** The window lengths for the antecedent (input) and consequent (output) parts of the model are defined. The total window length is 50, the antecedent part is 40, and the consequent part is the difference between the two (10).

```python
wtr = 50  # total window length
wte = 40  # window length for the antecedent part
wlm = wtr - wte  # window length for the consequent part
```
**7.3) Initialize MinMaxScaler:** The MinMaxScaler from sklearn is initialized. This scaler will be used to normalize the data.

```python
scaler = MinMaxScaler()
```
**7.4) Create and Normalize Antecedent Parts for Training Data:** The antecedent parts for the training data are created and normalized. This is done by creating sliding windows of length wtr over the training data, and then normalizing the first wte elements of each window. The normalized antecedent part and the non-normalized consequent part are then concatenated to form the final window.

```python
training_windows = [training_data[i:i + wtr] for i in range(len(training_data) - wtr + 1)]
training_windows_antece_normalized = [np.concatenate([scaler.fit_transform(window.iloc[:wte]), window.iloc[wte:wtr]], axis=0) for window in training_windows]
```
**7.5) Print Number of Training Windows:** Finally, the number of training windows is printed.

```python
print(f"Number of training windows: {len(training_windows)}")
output: Number of training windows: 2595
```

## 8) K-Means Clustering
K-means is a popular unsupervised learning algorithm used for clustering. The goal of K-means is to group data points into distinct non-overlapping subgroups. One of the major application of K-means is segmentation of data.

In your project, K-means clustering is advantageous as it groups similar stock market trends, represented by 'windows', into clusters. This allows for the identification of common patterns in stock behavior. Each cluster is then labeled as 'UP' or 'DOWN' using linear regression, indicating the overall trend of the stocks within that cluster. This information can be used to make informed predictions about future stock market trends, aiding in decision-making for investments and trading. More insights and detailed analysis will be provided in the upcoming sections of the project.

**8.1) Python implementatio of K-means clustering algorithm:**
This Python function, k_means_clustering, is an implementation of the K-means clustering algorithm. The function takes four parameters: data (the dataset to be clustered), k (the number of clusters), max_iterations (the maximum number of iterations to run the algorithm), and random_state (a seed for the random number generator to ensure reproducibility).

```python
def k_means_clustering(data, k, max_iterations=500, random_state=0):
    np.random.seed(random_state)

    # Initialize centroids randomly from the data points
    centroids = data[np.random.choice(len(data), size=k, replace=False)]
    
    for _ in range(max_iterations):
        # Assign each data point to the nearest cluster center
        distances = np.linalg.norm(data.reshape(len(data), -1)[:, np.newaxis] - centroids.reshape(len(centroids), -1), axis=2)
        labels = np.argmin(distances, axis=1)

        # Recalculate the centroids as the mean of the current clusters
        new_centroids = np.empty_like(centroids)
        empty_clusters = 0
        for i in range(k):
            members = data[labels == i]
            if len(members) > 0:
                new_centroids[i] = np.mean(members, axis=0)
            else:
                new_centroids[i] = centroids[i]
                #new_centroids[i] = data[np.random.randint(len(data))]
                empty_clusters += 1

        # Check for convergence
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids
```
The k_means_clustering function begins by initializing the centroids, randomly selecting k data points from the dataset to serve as the initial centroids. The main part of the function is a loop that runs for a maximum of max_iterations iterations. In each iteration, the function calculates the Euclidean distance between each data point and each centroid, then assigns each data point to the cluster whose centroid is nearest. The function then recalculates the centroids, initializing an array to hold the new centroids and calculating the new centroid for each cluster. If a cluster has no data points, its centroid remains the same, otherwise, the new centroid is the mean of the data points in the cluster. The function also keeps track of the number of empty clusters, incrementing a counter, empty_clusters, if a cluster has no data points. After recalculating the centroids, the function checks for convergence. If the centroids have not changed from the previous iteration, the algorithm has converged, and the function breaks out of the loop. Otherwise, it updates the centroids with the new values and proceeds to the next iteration. The function returns the final centroids and the labels of the clusters for each data point, which can be used to understand the clustering of the data and to make predictions for new data points.

**8.2) Kronecker delta function:**
This function is part of a clustering algorithm, likely K-means. It calculates the Kronecker delta function, used to indicate cluster membership of data points.

```python
# Calculate Kronecker delta function
    delta = np.zeros((len(data), k))
    for i in range(len(data)):
        delta[i, labels[i]] = 1

    return labels, centroids, delta
```
A two-dimensional numpy array delta is created, with dimensions equal to the number of data points and clusters. The code then iterates over each data point, marking its assigned cluster in the delta array. The function returns labels (the cluster assignment for each data point), centroids (the final cluster centers), and delta (a binary matrix indicating cluster membership).

In this project, the Kronecker delta function is used in the "Labelling the Clusters" section, where its role and usage will be further explained.

**8.3) Applying K-Means Clustering to Antecedent Windows:**
Now, let's apply the K-means clustering algorithm to the antecedent windows extracted from the training data. We'll use the k_means_clustering function we defined earlier to perform this task.and we are going to set number of clusters ```k=50```. Here's how we call the function:

```python
# Extract the antecedent part from each window
antecedent_windows = [window[:wte] for window in training_windows_antece_normalized]

# Run K-means clustering on the antecedent windows
labels, centroids, delta = k_means_clustering(np.array(antecedent_windows), k)
```
This code snippet extracts the antecedent part from each window in the training dataset and then applies the K-means clustering algorithm to these antecedent windows. The resulting labels, centroids, and delta matrix will be used for further analysis and labeling of the clusters.

## 9) Labeling the Clusters
In this section of the project code, we're assigning trend labels to the clusters generated by the K-means clustering algorithm. The goal is to categorize each cluster as exhibiting an upward ('UP') or downward ('DOWN') trend based on the behavior of the stock market data within that cluster.

**9.1) Initializing the Model and Trend Labels:**
To determine the trend for each cluster, we start by initializing a linear regression model and a dictionary to store the trend labels for each cluster

```python
# Initialize the Linear Regression model
model = LinearRegression()

# Initialize the trend labels
cluster_trends = {}
```
**9.2) Preparing the Data:**
Next, we convert the list training_windows_antece_normalized to a NumPy array and flatten it to 2D for easier manipulation.

```python
# Convert the list to a numpy array and flatten it to 2D
data_array = np.array(training_windows_antece_normalized).reshape(len(training_windows_antece_normalized), -1)
```
**9.3) Selecting Windows for Each Cluster:**
Using the delta matrix, calculated earlier with the Kronecker delta function, we select the windows belonging to each cluster. The condition delta[:, j] == 1 ensures we get the windows associated with cluster j.

```python
# For each cluster
for j in range(k):
    # Get the windows that belong to the cluster
    windows = data_array[delta[:, j] == 1]
    
    # If the cluster has no windows, skip it
    if len(windows) == 0:
        continue
```
**9.4) Extracting and Combining Consequent Parts:**
For each window in the cluster, the consequent part (following the antecedent part) is extracted and stored in a list. These consequent parts are then concatenated to form a single array, which is used to fit the linear regression model.

```python
    # Initialize an empty list to store the consequent parts
    consequent_parts = []
    
    # For each window in the cluster
    for window in windows:
        # Get the consequent part of the window
        consequent_part = window[wte:]
        
        # Add the consequent part to the list
        consequent_parts.append(consequent_part)
    
    # Combine the consequent parts of all windows in the cluster
    combined_consequent_part = np.concatenate(consequent_parts)
```
**9.5) Fitting the Linear Regression Model:**
The independent variable t, representing time steps, is created as an array of indices corresponding to the combined consequent parts. The model is then fitted to this data, and the slope of the fitted model indicates the trend.

```python
    # Fit the linear regression model
    t = np.arange(len(combined_consequent_part)).reshape(-1, 1)  # This is t
    model.fit(t, combined_consequent_part)  # This calculates a and b

    # Get the slope of the model
    slope = model.coef_[0]  # This is b
```
**9.6) Assigning Trend Labels:**
A positive slope suggests an 'UP' trend, while a negative slope indicates a 'DOWN' trend. This trend label is stored in the cluster_trends dictionary.

```python
    # Assign the trend label based on the slope
    if slope > 0:
        cluster_trends[j] = "UP"
    else:
        cluster_trends[j] = "DOWN"
```
**9.7) Counting and Printing Trend Labels:**
Finally, the number of 'UP' and 'DOWN' labels in the cluster_trends dictionary is counted and printed. By labeling the clusters in this manner, we identify the overall trend within each cluster based on the stock market behavior in the consequent parts of the windows. This labeling is essential for making predictions about future stock market trends and will be used in subsequent sections to label the test data and make trading decisions based on the predicted trends.

```python
# Count the number of "UP" labels
num_up = list(cluster_trends.values()).count("UP")

# Count the number of "DOWN" labels
num_down = list(cluster_trends.values()).count("DOWN")

# Print the counts
print("Number of 'UP' labels:", num_up)
print("Number of 'DOWN' labels:", num_down)

output: Number of 'UP' labels: 50
        Number of 'DOWN' labels: 0

```

## 10) Labeling the Test Data Windows
In this section, we label the test data windows based on the clusters generated from the training data. The goal is to predict whether each test window shows an upward ('UP') or downward ('DOWN') trend by comparing it to the trained clusters

**10.1) Setting Up and Creating Test Windows:**
We start by setting the size of the antecedent part of each window and creating a separate DataFrame for the test dates.

```python
antecedent_size = 40

# Create a separate DataFrame for the test dates
test_dates = dates[dates['Date'].dt.year > 2020].copy()
```
Next, we create windows for the test data, ensuring each window has the same antecedent size.

```python
# Create windows for the test data
test_windows = [test_data[i:i + antecedent_size] for i in range(len(test_data) - antecedent_size + 1)]
```
**10.2) Normalizing Test Windows:**
To ensure consistency in the data, we normalize each test window using the scaler fitted on the training data.

```python
# Normalize each window
test_windows_normalized = [scaler.transform(window) for window in test_windows]

# Print the total number of windows
print(f"Total number of windows: {len(test_windows)}")

output: Total number of windows: 815
```
**10.3) Initializing Trend Labels for Test Data:**
We initialize an empty list to store the trend labels for the test data.

```python
# Initialize the trend labels for the test data
test_trend_labels = []
```
**10.4) Labeling Each Test Window:**
For each normalized test window, we calculate the Euclidean distance to each cluster center to determine the closest cluster.

```python
# For each window in the test data
for window in test_windows_normalized:
    
    # Calculate the Euclidean distance to each cluster center
    distances = np.linalg.norm(centroids - window, axis=1)

    # Find the index of the closest cluster center
    closest_cluster = np.argmin(distances)
```
**10.5) Assigning Trend Labels:**
We check if the closest cluster index is valid and assign the corresponding trend label from the cluster_trends dictionary. If the index is invalid, an error message is printed.

```python
    # Check if the closest cluster index is a valid index for the labels array
    if closest_cluster < len(labels):
        # Assign the trend label of the closest cluster to the window
        test_trend_labels.append(cluster_trends[labels[closest_cluster]])
    else:
        # If the closest cluster index is not a valid index for the labels array, print an error message
        print(f"Error: closest cluster index {closest_cluster} is not a valid index for the labels array.")
```
**10.6) Counting and Printing Trend Labels:**
To summarize the results, we count the number of each trend label in the test_trend_labels list using the Counter class from the collections module.

```python
# Import the Counter class from the collections module
from collections import Counter

# Count the number of each label in test_trend_labels
label_counts = Counter(test_trend_labels)

# Print the counts
print(label_counts)

Total number of windows: 798
Counter({'UP': 798})
```
By labeling the test data windows, we predict the trend for each window based on its similarity to the clusters identified in the training data. This process helps in making informed decisions about stock market behavior, providing a basis for further analysis and trading strategies.

## 11) Trading Strategy

In this section, we implement a trading strategy based on the trend labels assigned to the test data windows. The goal is to simulate a trading portfolio, buying and selling shares based on predicted trends, and evaluate the performance over a series of years.

**11.1) Initial Setup:**
We start by defining initial parameters, including the trading cost and the years for which we will evaluate the trading strategy.

```python
# Define the cost
c = 0.00135

# Define the years
years = [2021, 2022, 2023, 2024]
```
The trading cost, denoted as c, is set to 0.00135. This cost represents the transaction fee associated with each trade, expressed as a proportion of the trade value. Trading costs are a standard aspect of stock trading, incurred during both buy and sell transactions. Including this cost in our simulation ensures that our trading strategy reflects real-world conditions more accurately.

We also initialize a list to store the final value of the portfolio for each year.

```python
# Initialize a list to store the final value of your portfolio for each year
final_values = []
```
**11.2) Simulating Trading for Each Year:**
For each year, we simulate trading by initializing the starting amount of money and the number of shares owned.

```python
# For each year
for year in years:
    # Initialize the amount of money you have and the number of shares you own
    money = 1000000.0
    shares = 0

    # Initialize a list to store trading actions
    actions = []
```
We then identify the indices of the test windows that correspond to the current year.

```python
    # Get the indices for the current year
    indices_year = [i for i, date in enumerate(test_dates['Date']) if date.year == year and i < len(test_windows)]
```
**11.3) Retrieving Test Windows and Trend Labels:**
We retrieve the test windows and trend labels for the current year based on the identified indices.

```python
    # Get the test windows and trend labels for the current year
    test_windows_year = [test_windows[i] for i in indices_year]
    test_trend_labels_year = [test_trend_labels[i] for i in indices_year]
```
**11.4) Executing Trades Based on Trend Labels:**
For each window in the test data for the current year, we calculate the current stock price and determine the trading action based on the trend label

```python
    # For each window in the test data for the current year
    for i in range(len(test_windows_year)):
        # Calculate the current stock price
        current_price = test_windows_year[i].iloc[0]['Adj Close']

        # Check the trend label for the current window
        current_trend_label = test_trend_labels_year[i]
```
If the trend label is "UP", we buy shares at the beginning of the next window, investing 25% of the available money.

```python
        if current_trend_label == "UP" :
            # If the current window is "UP", buy shares at the beginning of the next window
            if i + 1 < len(test_windows_year):
                next_price = test_windows_year[i + 1].iloc[0]['Adj Close']
                shares_to_buy = math.floor((0.25 * money) / (next_price * (1 + c)))
                money -= shares_to_buy * next_price * (1 + c)
                shares += shares_to_buy
                actions.append("Buy")
```
If the trend label is "DOWN" and we own shares, we sell all shares at the beginning of the next window.

```python
        elif current_trend_label == "DOWN" and shares > 0:
            # If the current window is "DOWN", sell all shares at the beginning of the next window
            if i + 1 < len(test_windows_year):
                next_price = test_windows_year[i + 1].iloc[0]['Adj Close']
                money += shares * next_price * (1 - c)
                shares = 0
                actions.append("Sell")
```
**11.5) Finalizing the Year-End Portfolio Value:**
At the end of the year, we sell any remaining shares and calculate the final value of the portfolio.

```python
    # Sell any shares left at the end of the last day
    if shares > 0:
        sell_price = test_windows_year[-1].iloc[-1]['Adj Close']
        money += shares * sell_price * (1 - c)
        shares = 0
        actions.append("Sell")

    # Calculate the final value of your portfolio
    final_value = money

    # Store the final value
    final_values.append(final_value)
```
**11.6) Evaluating and Printing the Results:**
We calculate and print the return for each year, as well as the number of buy and sell actions executed.

```python
    # Calculate the return
    return_percentage = (final_value / 1000000.0 - 1) * 100

    # Print the return
    print(f"Return for {year}: {return_percentage}%")

    # Count the number of each action
    buy_count = actions.count("Buy")
    sell_count = actions.count("Sell")

    # Print the counts
    print(f"Number of Buy actions in {year}: {buy_count}")
    print(f"Number of Sell actions in {year}: {sell_count}")
```
I tested the code on 5 major stock exchanges in the world. The ongoing example uses the ^NSEI (Nifty 50 index of India), and the returns for this index are as follows:
```
^NSEI:
Return for 2021: 15.852735476040047%
Number of Buy actions in 2021: 247
Number of Sell actions in 2021: 1
Return for 2022: -2.485190345751953%
Number of Buy actions in 2022: 247
Number of Sell actions in 2022: 1
Return for 2023: 20.92553105589845%
Number of Buy actions in 2023: 244
Number of Sell actions in 2023: 1
Return for 2024: 5.400270834062515%
Number of Buy actions in 2024: 56
Number of Sell actions in 2024: 1
```
Similarly, I tested the strategy on the following four other major stock exchanges:

```
^SPX (S&P 500 Index):
Return for 2021: 0.6085992972644894%
Number of Buy actions in 2021: 146
Number of Sell actions in 2021: 17
Return for 2022: -17.19516062675408%
Number of Buy actions in 2022: 152
Number of Sell actions in 2022: 22
Return for 2023: 20.96196201629159%
Number of Buy actions in 2023: 145
Number of Sell actions in 2023: 18
Return for 2024: 1.9384591040576327%
Number of Buy actions in 2024: 31
Number of Sell actions in 2024: 5
```
```
^FTSE (FTSE 100 Index):
Return for 2021: -5.3988042074120335%
Number of Buy actions in 2021: 117
Number of Sell actions in 2021: 21
Return for 2022: -0.9240033206517562%
Number of Buy actions in 2022: 159
Number of Sell actions in 2022: 17
Return for 2023: -5.708509080065793%
Number of Buy actions in 2023: 131
Number of Sell actions in 2023: 18
Return for 2024: 5.653056667255907%
Number of Buy actions in 2024: 38
Number of Sell actions in 2024: 5
```
```
^HSI (Hang Seng Index):
Return for 2021: -17.89278680816404%
Number of Buy actions in 2021: 236
Number of Sell actions in 2021: 4
Return for 2022: -17.696382923300767%
Number of Buy actions in 2022: 244
Number of Sell actions in 2022: 2
Return for 2023: -19.393371522597647%
Number of Buy actions in 2023: 242
Number of Sell actions in 2023: 1
Return for 2024: 13.261341658862325%
Number of Buy actions in 2024: 56
Number of Sell actions in 2024: 2
```
```
^N225 (Nikkei 225 Index):
Return for 2021: -0.714269791054678%
Number of Buy actions in 2021: 5
Number of Sell actions in 2021: 3
Return for 2022: 0.44509055836916556%
Number of Buy actions in 2022: 10
Number of Sell actions in 2022: 5
Return for 2023: 1.4126984671777754%
Number of Buy actions in 2023: 27
Number of Sell actions in 2023: 9
Return for 2024: 2.2763819915039063%
Number of Buy actions in 2024: 7
Number of Sell actions in 2024: 2
```

Our project's results demonstrate the potential of our chosen strategy across multiple major stock exchanges over a four-year period. The strategy yielded positive returns in several instances, notably a 20.93% return for the NSEI in 2023 and a 20.96% return for the S&P 500 Index in the same year. These results highlight the strategy's potential effectiveness in certain market conditions.

However, it's important to note that the strategy did not consistently yield positive results across all markets and years. For instance, the Hang Seng Index saw negative returns for three consecutive years from 2021 to 2023. This underscores the inherent risks and variability in stock market investments, and the need for robust risk management strategies.

It's also worth noting that our project assumed a constant transaction cost across all markets. In reality, transaction costs can vary significantly between markets and over time. This simplification may have influenced the results and should be considered when interpreting the project's findings.

## 12) Evaluation with Sharpe Ratio

In this section, we introduce the Sharpe Ratio as an additional evaluation metric to assess the risk-adjusted performance of our trading strategy. The Sharpe Ratio compares the average excess return (returns above the risk-free rate) to the standard deviation of those excess returns.

```python
# Risk-free rate
risk_free_rate = 0.02  # Annual risk-free rate (2%)

# Calculate the annual returns
returns = [(final_values[i] / final_values[i-1]) - 1 for i in range(1, len(final_values))]

# Calculate the excess returns (subtracting the risk-free rate from each return)
excess_returns = [r - risk_free_rate for r in returns]

# Calculate the average excess return
average_excess_return = np.mean(excess_returns)

# Calculate the standard deviation of the excess returns
standard_deviation = np.std(excess_returns)

# Calculate the Sharpe Ratio
sharpe_ratio = average_excess_return / standard_deviation

print(f"Sharpe Ratio: {sharpe_ratio}")
```
In the initial setup, we define the risk-free rate as 2%, representing the annual risk-free rate assumed for our evaluation. We then compute the annual returns of our trading strategy based on the final portfolio values across the specified years. Using these returns, we calculate the excess returns by subtracting the risk-free rate from each return. Next, we determine the average excess return and the standard deviation of the excess returns. Finally, we compute the Sharpe Ratio by dividing the average excess return by the standard deviation, providing a measure of risk-adjusted performance for our trading strategy.

**Sharpe Ratio for Each Stock Market:**

- ^NSEI: -0.19707167490114547
- ^SPX: 0.07494709503820274
- ^FTSE: 0.2870109954250266
- ^HSI: 0.5573678534562041
- ^N225: -7.685668619341495

The Sharpe Ratios calculated for each stock market provide insights into the risk-adjusted performance of the trading strategy. A positive Sharpe Ratio indicates that the strategy has yielded returns above the risk-free rate, while a negative Sharpe Ratio suggests the strategy has underperformed the risk-free rate.

The strategy performed best on the Hang Seng Index (^HSI) with a Sharpe Ratio of 0.557, followed by the FTSE 100 Index (^FTSE) with a ratio of 0.287. These positive Sharpe Ratios suggest that the strategy has been effective in these markets, yielding returns that exceed the risk-free rate after adjusting for risk.

However, the strategy underperformed on the NSEI and N225, with Sharpe Ratios of -0.197 and -7.685 respectively. This suggests that the strategy's returns did not compensate for the risk taken in these markets.

It's important to note that these results are based on an assumed risk-free rate of 2% across all markets. In reality, risk-free rates can vary significantly between countries due to factors such as differences in interest rates set by central banks, inflation rates, and economic conditions. This simplification may have influenced the results and should be considered when interpreting the project's findings.

## Conclusion

In conclusion, while the strategy demonstrated promising results in some markets, it also highlighted the complexities and challenges of stock market investment. Future work could explore varying risk-free rates, incorporating more sophisticated risk management strategies, and testing the approach on a wider range of markets.
















    
















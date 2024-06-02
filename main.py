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
from collections import Counter

def main():
    # DATA PREPROCESSING

    # Download stock price data
    data = yf.download("^NSEI", start="2010-01-01", end="2024-05-25", interval="1d")

    # Calculate RSI
    rsi_indicator = RSIIndicator(close=data['Adj Close'], window=14)
    data['RSI'] = rsi_indicator.rsi()

    # Calculate 50-day EMA
    ema_indicator = EMAIndicator(close=data['Adj Close'], window=50)
    data['50_EMA'] = ema_indicator.ema_indicator()
    data.dropna(inplace=True)

    # Drop the 'Close' column
    data = data.drop(columns=['Close'])

    # Store dates in a separate DataFrame
    dates = data.index.to_frame(index=False)

    # Reset index
    data.reset_index(drop=True, inplace=True)

    # Divide the data into training and test data based on the date
    training_data = data[dates['Date'].dt.year <= 2020].copy()
    #print(training_data.head(5))
    test_data = data[dates['Date'].dt.year > 2020].copy()

    # Define the window lengths for the antecedent and consequent parts
    wtr = 50  # total window length
    wte = 40  # window length for the antecedent part
    wlm = wtr - wte  # window length for the consequent part

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Create and normalize the antecedent parts for the training data
    training_windows = [training_data[i:i + wtr] for i in range(len(training_data) - wtr + 1)]
    training_windows_antece_normalized = [np.concatenate([scaler.fit_transform(window.iloc[:wte]), window.iloc[wte:wtr]], axis=0) for window in training_windows]

    # Print the number of windows
    print(f"Number of training windows: {len(training_windows)}")

    # K- MEANS CLUSTERING

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

        print(f"Number of empty clusters: {empty_clusters}")

        # Calculate Kronecker delta function
        delta = np.zeros((len(data), k))
        for i in range(len(data)):
            delta[i, labels[i]] = 1

        return labels, centroids, delta

    k = 50 # number of clusters

    # Extract the antecedent part from each window
    antecedent_windows = [window[:wte] for window in training_windows_antece_normalized]

    # Run K-means clustering on the antecedent windows
    labels, centroids, delta = k_means_clustering(np.array(antecedent_windows), k)

    # LABELLING THE CLUSTERS 
 
    # Initialize the Linear Regression model
    model = LinearRegression()

    # Initialize the trend labels
    cluster_trends = {}

    # Convert the list to a numpy array and flatten it to 2D
    data_array = np.array(training_windows_antece_normalized).reshape(len(training_windows_antece_normalized), -1)

    # For each cluster
    for j in range(k):
        # Get the windows that belong to the cluster
    
        windows = data_array[delta[:, j] == 1]
        
        # If the cluster has no windows, skip it
        if len(windows) == 0:
            continue
    
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
        
        # Fit the linear regression model
        t = np.arange(len(combined_consequent_part)).reshape(-1, 1)  # This is t
        model.fit(t, combined_consequent_part)  # This calculates a and b

        # Get the slope of the model
        slope = model.coef_[0]  # This is b

        # Assign the trend label based on the slope
        if slope > 0:
            cluster_trends[j] = "UP"
        else:
            cluster_trends[j] = "DOWN"

    # Count the number of "UP" labels
    num_up = list(cluster_trends.values()).count("UP")

    # Count the number of "DOWN" labels
    num_down = list(cluster_trends.values()).count("DOWN")

    # Print the counts
    print("Number of 'UP' labels:", num_up)
    print("Number of 'DOWN' labels:", num_down)

    # LABELLING THE TEST DATA WINDOWS

    antecedent_size = 40

    # Create a separate DataFrame for the test dates
    test_dates = dates[dates['Date'].dt.year > 2020].copy()

    # Create windows for the test data
    test_windows = [test_data[i:i + antecedent_size] for i in range(len(test_data) - antecedent_size + 1)]

    # Normalize each window
    test_windows_normalized = [scaler.transform(window) for window in test_windows]

    # Print the total number of windows
    print(f"Total number of windows: {len(test_windows)}")

    # Initialize the trend labels for the test data
    test_trend_labels = []

    # For each window in the test data
    for window in test_windows_normalized:
        
        # Calculate the Euclidean distance to each cluster center
        distances = np.linalg.norm(centroids - window, axis=1)

        # Find the index of the closest cluster center
        closest_cluster = np.argmin(distances)

        # Check if the closest cluster index is a valid index for the labels array
        if closest_cluster < len(labels):
            # Assign the trend label of the closest cluster to the window
            test_trend_labels.append(cluster_trends[labels[closest_cluster]])
        else:
            # If the closest cluster index is not a valid index for the labels array, print an error message
            print(f"Error: closest cluster index {closest_cluster} is not a valid index for the labels array.")

    # Import the Counter class from the collections module
    from collections import Counter

    # Count the number of each label in test_trend_labels
    label_counts = Counter(test_trend_labels)

    # Print the counts
    print(label_counts)

    # TRADNG STRATEGY

    # Define the cost
    c = 0.00135

    # Define the years
    years = [2021, 2022, 2023, 2024]

    # Initialize a list to store the final value of your portfolio for each year
    final_values = []

    # For each year
    for year in years:
        # Initialize the amount of money you have and the number of shares you own
        money = 1000000.0
        shares = 0

        # Initialize a list to store trading actions
        actions = []

        # Get the indices for the current year
        indices_year = [i for i, date in enumerate(test_dates['Date']) if date.year == year and i < len(test_windows)]
        

        # Get the test windows and trend labels for the current year
        test_windows_year = [test_windows[i] for i in indices_year]
        test_trend_labels_year = [test_trend_labels[i] for i in indices_year]

        # For each window in the test data for the current year
        for i in range(len(test_windows_year)):
            # Calculate the current stock price
            current_price = test_windows_year[i].iloc[0]['Adj Close']

            # Check the trend label for the current window
            current_trend_label = test_trend_labels_year[i]

            if current_trend_label == "UP" :
                # If the current window is "UP", buy shares at the beginning of the next window
                if i + 1 < len(test_windows_year):
                    next_price = test_windows_year[i + 1].iloc[0]['Adj Close']
                    shares_to_buy = math.floor((0.25 * money) / (next_price * (1 + c)))
                    money -= shares_to_buy * next_price * (1 + c)
                    shares += shares_to_buy
                    actions.append("Buy")

            elif current_trend_label == "DOWN" and shares > 0:
                # If the current window is "DOWN", sell all shares at the beginning of the next window
                if i + 1 < len(test_windows_year):
                    next_price = test_windows_year[i + 1].iloc[0]['Adj Close']
                    money += shares * next_price * (1 - c)
                    shares = 0
                    actions.append("Sell")

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

        # Calculate the return
        return_percentage = (final_value / 1000000.0 - 1) * 100

        # Print the return
        print(f"Return for {year}: {return_percentage}%")

        # Count the number of each action
        buy_count = actions.count("Buy")
        sell_count = actions.count("Sell")
        #stay_count = actions.count("Stay")

        # Print the counts
        print(f"Number of Buy actions in {year}: {buy_count}")
        print(f"Number of Sell actions in {year}: {sell_count}")

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
 


if __name__ == "__main__":
    main()
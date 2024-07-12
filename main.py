import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Ask the user for the company ticker
company = input("Enter the company ticker symbol: ")

try:
    # Date range for training data (adjust as needed)
    start_train = dt.datetime(2012, 1, 1)
    end_train = dt.datetime.now()  # Use current date for training end date

    # Load data using yfinance
    data = yf.download(company, start=start_train, end=end_train)

    if data.empty:
        raise ValueError(f"No data found for {company}. Please check the company ticker or try again later.")

    # Prepare data for training
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Load recent test data
    test_start = end_train - dt.timedelta(days=365)  # Use data from the last year for testing
    test_end = end_train

    test_data = yf.download(company, start=test_start, end=test_end)

    # Checks if test data is empty
    if test_data.empty:
        raise ValueError(f"No recent test data found for {company}. Please check the company ticker or try again later.")

    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make predictions on test data
    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot the test predictions
    plt.plot(test_data.index, actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(test_data.index, predicted_prices, color="green", label=f"Predicted {company} Price")
    plt.title(f"{company} Share Price Prediction")
    plt.xlabel('Time')
    plt.ylabel(f"{company} Share Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Predicting the next day
    real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction for the next day: {prediction[0][0]}")

except Exception as e:
    print(f"Error occurred: {e}")

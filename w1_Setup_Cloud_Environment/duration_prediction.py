#!/usr/bin/env python
"""
Duration Prediction - Basic Implementation

Basic implementation of NYC taxi trip duration prediction for learning purposes.

Author: Daniel Lachner-Piza
Email: dalapiz@proton.me
"""

import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, r2_score


def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df



# Load the dataset
df = pd.read_parquet('~/Data/green_tripdata_2021-01.parquet')

# Subtract the pickup and dropoff times, this will give a Timedelta, which has a total_seconds method
df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
# Convert the Timedelta to minutes applying the total_seconds method to each row
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

# Filter out rows with invalid durations
df = df[(df.duration >= 1) & (df.duration <= 60)]


# Selec feature by data type
categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']

# Convert categorical features to string type
#df[categorical] = df[categorical].map(str)
df[categorical] = df[categorical].astype(str)

# Transform the mixed data types into a format suitable for machine learning using DictVectorizer
# DictVectorizer converts a list of dictionaries into a matrix of features
# Each dictionary corresponds to a row in the DataFrame, with keys as feature names and values as feature values
# NUmerical features are already in a suitable format, while categorical features need to be one-hot encoded
# The DictVectorizer will handle this automatically by creating binary columns for each category in the categorical features
# The resulting matrix will have one column for each unique value in the categorical features and one column for each numerical feature
# The resulting matrix will be sparse, meaning it will only store non-zero values to save memory
# The DictVectorizer will also handle missing values by treating them as a separate category  


# Convert the DataFrame to a list of dictionaries, where each dictionary represents a row
train_dicts = df[numerical+categorical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

target = 'duration'
y_train = df[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

rmse_val = root_mean_squared_error(y_train, y_pred)
r2_val = r2_score(y_train, y_pred)
print(f'RMSE: {rmse_val:.3f}, R2: {r2_val:.3f}')

sns.displot(y_pred, label='prediction')
sns.displot(y_train, label='actual')

plt.legend()
os.makedirs('images', exist_ok=True)
plt.savefig('images/duration_prediction_distribution.png')
# plt.show()
# plt.waitforbuttonpress()


pass


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_parquet('~/Data/green_tripdata_2021-01.parquet')
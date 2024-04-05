import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

financials = pd.read_csv(r"data\financials.csv")

print(financials.head())

financialsReduced = financials[['Units Sold', 'Sale Price']]

print(financialsReduced.head())

train, val, test = np.split(financialsReduced.sample(frac=1), [int(0.8*len(financialsReduced)), int(0.9*len(financialsReduced))])

print(len(train), 'training examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

def df_to_dataset(financialsReduced, shuffle=True, batch_size=32):
  df = financialsReduced.copy()
  labels = df.pop('Sale Price')
  df = {key: value.values[:,tf.newaxis] for key, value in financialsReduced.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(financialsReduced))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)

[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of units sold:', train_features['Units Sold'])
print('A batch of targets:', label_batch )

def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

unitsSoldCol = train_features['Units Sold']
layer = get_normalization_layer('Units Sold', train_ds)
layer(unitsSoldCol)

























# print(salesArrayAnalysis)

# salesModel = tf.keras.Sequential([
#   layers.Dense(64, activation='relu'),
#   layers.Dense(1)
# ])

# salesModel.compile(loss = tf.keras.losses.MeanSquaredError(),
#                       optimizer = tf.keras.optimizers.Adam())

# salesModel.fit(salesArrayAnalysis, salesDataLabels, epochs=10)
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

financials = pd.read_csv(r"data\financials.csv")

print(financials.head())

salesAnalysis = financials.copy()
salesDataLabels = financials.pop('Country')

salesArrayAnalysis = np.array(salesAnalysis)

print(salesArrayAnalysis)

# salesModel = tf.keras.Sequential([
#   layers.Dense(64, activation='relu'),
#   layers.Dense(1)
# ])

# salesModel.compile(loss = tf.keras.losses.MeanSquaredError(),
#                       optimizer = tf.keras.optimizers.Adam())

# salesModel.fit(salesArrayAnalysis, salesDataLabels, epochs=10)
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

financials = pd.read_csv(r"data\financialsSimple.csv")

print(financials.head())

financialsFeatures = np.array(financials[['Sale Price']])
financialsLabels = np.array(financials[['Units Sold']])

financialsModel = tf.keras.Sequential([
  layers.Dense(64, activation='relu'),
  layers.Dense(1)
])

financialsModel.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

financialsModel.fit(financialsFeatures, financialsLabels, epochs=10)
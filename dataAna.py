import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

df = pd.read_csv(r"data\financials.csv")

print(df)

# import tensorflow as tf
# from tensorflow.keras import layers


import pandas
import numpy as np
import tensorflow as tf

# download data
dataframe = pandas.read_csv('https://raw.githubusercontent.com/bryankolaczkowski/ALS3200C/main/mbiome.data.csv')

# create train-validate split
train_dataframe = dataframe.sample(frac=0.8, random_state=2100963)
valid_dataframe = dataframe.drop(train_dataframe.index)

# extract explanatory variables
dta_ids = [ x for x in dataframe.columns if x.find('DTA') == 0 ]
train_x = np.expand_dims(train_dataframe[dta_ids].to_numpy(), axis=-1)
valid_x = np.expand_dims(valid_dataframe[dta_ids].to_numpy(), axis=-1)

# extract labels
train_y = train_dataframe['LBL0'].to_numpy()
valid_y = valid_dataframe['LBL0'].to_numpy()

# package data into tensorflow dataset
train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32)
valid_data = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(32)

# build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=16, return_sequences=True, input_shape=(256,1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid))
model.summary()


# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# fit model
model.fit(train_data, epochs=50, validation_data=valid_data)

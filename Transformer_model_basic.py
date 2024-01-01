import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# download data
dataframe = pd.read_csv('https://raw.githubusercontent.com/bryankolaczkowski/ALS3200C/main/mbiome.data.csv')

# create train-validate split
train_dataframe = dataframe.sample(frac=0.8, random_state=827847)
valid_dataframe = dataframe.drop(train_dataframe.index)

# extract explanatory variables
dta_ids = [ x for x in dataframe.columns if x.find('DTA') == 0 ]
train_x = train_dataframe[dta_ids].to_numpy()
valid_x = valid_dataframe[dta_ids].to_numpy()

# add 'location' to sequence data
loc = np.linspace(start=-2.5, stop=+2.5, num=train_x.shape[1])
train_x = np.stack([ train_x, np.array([loc]*train_x.shape[0]) ], axis=-1)
valid_x = np.stack([ valid_x, np.array([loc]*valid_x.shape[0]) ], axis=-1)

# extract labels
train_y = train_dataframe['LBL0'].to_numpy()
valid_y = valid_dataframe['LBL0'].to_numpy()

# package data into tensorflow dataset
train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32)
valid_data = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(32)

# build model using functional api
repdim = 4 # set internal data representation dimensionality

# input and linear projection
inlayer = tf.keras.Input(shape=(256, 2))
proj = tf.keras.layers.Dense(units=repdim)(inlayer)

# multi-head attention block
mha1 = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=repdim)(proj, proj, proj)
res1 = tf.keras.layers.Add()([proj, mha1])
nrm1 = tf.keras.layers.LayerNormalization()(res1)

# feed-forward block
ffa1 = tf.keras.layers.Dense(units=repdim, activation=tf.keras.activations.relu)(nrm1)
ffb1 = tf.keras.layers.Dense(units=repdim)(ffa1)
res2 = tf.keras.layers.Add()([nrm1, ffb1])
nrm2 = tf.keras.layers.LayerNormalization()(res2)

# classification block
flt = tf.keras.layers.Flatten()(nrm2)
outlayer = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(flt)

model = tf.keras.Model(inputs=inlayer, outputs=outlayer)
model.summary()

# compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)

# fit model
model.fit(train_data, epochs=50, validation_data=valid_data)



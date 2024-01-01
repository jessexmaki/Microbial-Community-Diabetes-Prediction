import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import roc_auc_score

# Load your data
dataframe = pd.read_csv('https://raw.githubusercontent.com/bryankolaczkowski/ALS3200C/main/mbiome.data.csv')

# Extract explanatory variables and label
dta_ids = [x for x in dataframe.columns if x.startswith('DTA')]
X = dataframe[dta_ids].to_numpy()
y = dataframe['LBL0'].to_numpy()

# Add 'location' to sequence data
loc = np.linspace(start=-2.5, stop=+2.5, num=X.shape[1])
X = np.stack([X, np.array([loc] * X.shape[0])], axis=-1)

# Train-validation split
train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Package data into TensorFlow datasets
train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32)
valid_data = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(32)
# Define the model building function
def build_model():
    repdim = 32  # Set internal data representation dimensionality
    inlayer = tf.keras.Input(shape=(256, 2))
    proj = tf.keras.layers.Dense(units=repdim, kernel_regularizer=regularizers.l2(0.001))(inlayer)
    mha1 = tf.keras.layers.MultiHeadAttention(num_heads=14, key_dim=repdim)(proj, proj, proj)
    res1 = tf.keras.layers.Add()([proj, mha1])
    nrm1 = tf.keras.layers.LayerNormalization()(res1)
    mha2 = tf.keras.layers.MultiHeadAttention(num_heads=14, key_dim=repdim)(nrm1, nrm1, nrm1)
    res2 = tf.keras.layers.Add()([nrm1, mha2])
    nrm2 = tf.keras.layers.LayerNormalization()(res2)
    ffa1 = tf.keras.layers.Dense(units=repdim*4, activation='relu')(nrm2)
    ffb1 = tf.keras.layers.Dense(units=repdim, kernel_regularizer=regularizers.l2(0.001))(ffa1)
    res3 = tf.keras.layers.Add()([nrm2, ffb1])
    nrm3 = tf.keras.layers.LayerNormalization()(res3)
    ffa2 = tf.keras.layers.Dense(units=repdim*4, activation='relu')(nrm3)
    ffb2 = tf.keras.layers.Dense(units=repdim, kernel_regularizer=regularizers.l2(0.001))(ffa2)
    res5 = tf.keras.layers.Add()([nrm3, ffb2])
    nrm5 = tf.keras.layers.LayerNormalization()(res5)
    dropout_layer = tf.keras.layers.Dropout(0.4)(nrm5)
    flt = tf.keras.layers.Flatten()(dropout_layer)
    outlayer = tf.keras.layers.Dense(units=1, activation='sigmoid')(flt)
    model = tf.keras.Model(inputs=inlayer, outputs=outlayer)
    return model

# Build and compile the model
model = build_model()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Fit model on training data
history = model.fit(
    train_data,
    epochs=70,
    validation_data=valid_data,
    callbacks=[early_stopping]
)

# Model summary
model.summary()

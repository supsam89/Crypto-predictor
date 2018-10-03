import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization # we also want to normalise the batches
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint # Saves for certain parameters (e.g. Validation accuracy)

SEQ_LEN = 60 # length of data used in minutes
FUTURE_PERIOD_PREDICT = 3 # time we wish to predict ahead in minutes
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

# create the rule for the target values
def classify(current, future):
    if float(future) > float(current):
        return 1 # representing a buy
    else:
        return 0 # representing a sell

# create the preprocess function
def preprocess_df(df):
    # drop the future column in the original dataframe
    df = df.drop('future', 1)

    # we scale over the columns of different ratios (LTC, BTC etc..) (excluding the target column)
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            # drop any N/A values
            df.dropna(inplace=True)
            # scaling function
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    # deque appends to list the values until it hits the maxlen and then it restarts
    prev_days = deque(maxlen=SEQ_LEN)

    # iterate over the values in the dataframe
    for i in df.values:
        # iterate over each value in each column up to and not including target
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            # we append our x's and y's with a label of 0 or 1
            sequential_data.append([np.array(prev_days), i[-1]])

    # randomly shuffle the data
    random.shuffle(sequential_data)

    buys = []
    sells = []

    # add sells and buys to a list
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    # random shuffle (for good measure)
    random.shuffle(buys)
    random.shuffle(sells)

    # find what listt is lower
    lower = min(len(buys), len(sells))

    # balance the buys and sells
    buys = buys[:lower]
    sells = sells[:lower]

    # add the lists
    sequential_data = buys+sells
    random.shuffle(sequential_data)

    x = []
    y = []

    # create x and y values of data and targets
    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)

    return np.array(x), y

# create an empty dataframe
main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

for ratio in ratios:

    # state the dataset
    dataset = f"crypto_data/{ratio}.csv"

    # create column headers of original csv
    df = pd.read_csv("crypto_data/LTC-USD.csv", names=["time", "low", "high", "open", "close", "volume"])

    # rename the columns to the new dataframe (inplace=True so we do not have to redefine df)
    df.rename(columns={"close": f"{ratio}_close", "volume":f"{ratio}_volume"}, inplace=True)

    # set the index of the new dataframe
    df.set_index("time", inplace=True)

    # use the renamed columns
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    # merge the dataframes of all ratios
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

# create the dataframe for the future values
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT) # here we are shifting the original dataframe's close values up by the -FUTURE_PERIOD_PREDICT

# here we create that maps the classify function to the close and future column values
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

# sort the time values in order
times = sorted(main_df.index.values)

# select the last 5%
last_5pct = times[-int(0.05*len(times))]

# set the validation data to the last 5%
validation_main_df = main_df[(main_df.index >= last_5pct)]

# set the training data to the first 95%
main_df = main_df[(main_df.index < last_5pct)]

# run the prprocess on the training and validation data
train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Don't buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

# state the model
model = Sequential()
# use CuDNNLSTM for a GPU, 128 nodes
model.add(LSTM(128, input_shape=(train_x.shape[1:]),return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]),return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# this is a dense layer
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

# this is a optimizer selecting the learning rate and decay
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}" # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves the best ones

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint])

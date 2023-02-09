import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback, EarlyStopping

# from rrs_kit.DataClass import DataPath, VarSet
from DataClass import DataPath, VarSet

dp = DataPath()


def mews_sbp(sbp: float) -> int:
    score = 0
    # SBP score
    if sbp <= 70:
        score += 3
    elif sbp <= 80:
        score += 2
    elif sbp <= 100:
        score += 1
    elif sbp >= 200:
        score += 2

    return score


def mews_hr(hr: float) -> int:
    score = 0

    # hr
    if hr <= 40:
        score += 2
    elif hr <= 50:
        score += 1
    elif hr <= 100:
        score += 0
    elif hr <= 110:
        score += 1
    elif hr <= 130:
        score += 2
    else:
        score += 3

    return score


def mews_rr(rr: float) -> int:
    score = 0

    if rr <= 8:
        score += 2
    elif rr <= 14:
        score += 0
    elif rr <= 20:
        score += 1
    elif rr <= 29:
        score += 2
    else:
        score += 3

    return score


def mews_bt(bt: float) -> int:
    score = 0

    if bt <= 35:
        score += 1
    elif bt <= 38.4:
        score += 0
    else:
        score += 2

    return score


def mews(hr: float, rr: float, sbp: float, bt: float) -> int:
    s_hr = mews_hr(hr)
    s_rr = mews_rr(rr)
    s_sbp = mews_sbp(sbp)
    s_bt = mews_bt(bt)

    score = s_hr + s_rr + s_sbp + s_bt

    return score


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i : i + tw]
        train_label = input_data[i + tw : i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def train_valid_split(train_valid_seq, train_ratio=0.8):

    abn_id = np.unique(train_valid_seq.loc[train_valid_seq.is_abn == 1]["Patient"])
    nl_id = np.unique(train_valid_seq.loc[train_valid_seq.is_abn == 0]["Patient"])

    abn_train_id, abn_val_id = train_test_split(abn_id, train_size=train_ratio, random_state=716)
    nl_train_id, nl_val_id = train_test_split(nl_id, train_size=train_ratio, random_state=716)

    train_id = np.concatenate([nl_train_id, abn_train_id])
    valid_id = np.concatenate([nl_val_id, abn_val_id])

    train_valid_seq = train_valid_seq.set_index("Patient")

    train_id = np.sort(train_id)
    valid_id = np.sort(valid_id)

    train_seq = train_valid_seq.loc[train_id]
    valid_seq = train_valid_seq.loc[valid_id]

    return train_seq.reset_index(), valid_seq.reset_index()


def load_modeling_data(pickle_dir=dp.output_path):

    ## NOTICE ME
    with open(os.path.join(pickle_dir, "train_whole.pickle"), "rb") as f:
        train_seq = pickle.load(f)

    train_x = np.stack(train_seq.sequence.values, axis=0)
    train_y = np.stack(train_seq.target.values, axis=0).reshape(-1, 1)

    with open(os.path.join(pickle_dir, "test_whole.pickle"), "rb") as f:
        test_seq = pickle.load(f)

    valid_x = np.stack(test_seq.sequence.values, axis=0)
    valid_y = np.stack(test_seq.target.values, axis=0).reshape(-1, 1)

    return train_x, train_y, valid_x, valid_y


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def train_model(pickle_dir, model_name):

    scaler = MinMaxScaler(feature_range=(-1, 1))

    print("Loading Modeling Data ...")
    train_x, train_y, valid_x, valid_y = load_modeling_data(pickle_dir)

    train_x = scaler.fit_transform(train_x.reshape(-1, 1)).shape
    print("Model Training ...")
    with tf.device("/device:GPU:0"):
        model_GRU = Sequential()
        model_GRU.add(layers.BatchNormalization(input_shape=(None, 26)))
        model_GRU.add(layers.SimpleRNN(100, dropout=0.2, return_sequences=True))
        model_GRU.add(layers.SimpleRNN(50, dropout=0.2, return_sequences=True))
        model_GRU.add(layers.SimpleRNN(25, dropout=0.1))
        model_GRU.add(layers.Dense(1, activation="sigmoid"))
        model_GRU.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        my_callbacks = [EarlyStopping(monitor="val_loss", patience=20, verbose=2, mode="min")]
        model_GRU_json = model_GRU.to_json()

        with open(os.path.join(pickle_dir, model_name + ".json"), "w") as jf:
            jf.write(model_GRU_json)

        history_no_sampling = model_GRU.fit(
            train_x,
            train_y,
            epochs=1000,
            batch_size=200,
            validation_data=(valid_x, valid_y),
            shuffle=True,
            verbose=2,
            callbacks=my_callbacks,
        )

    history_no_sampling.model.save_weights(os.path.join(pickle_dir, model_name + ".h5"))


def load_and_eval_model(pickle_dir, model_name):

    print("Loaing Testing Data ...")
    _, _, _, _.test_x, test_y = load_modeling_data(pickle_dir)

    print("Loading Model ...")
    with open(f"{pickle_dir}/Model/{model_name}.json") as jf:
        loaded_model_json = jf.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(f"{pickle_dir}/Model/{model_name}.h5")

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("Predicting ...")
    y_pred_proba = model.predict_proba(test_x)

    y_pred = y_pred_proba.flatten()
    y_tf = test_y.flatten()
    plot_ROC_rev(y_tf, y_pred)
    plot_PR_curve(y_tf, y_pred)


def load_and_predict(pickle, model):
    print("Loading testing data...")

    test_x, test_y = load_modeling_data(pickle_dir, test_data=True)

    print("Loading Model ...")
    with open(os.path.join(model_path, model_name + ".json")) as f:
        loaded_model_json = f.read()
    # with open(f'{pickle_dir}/Model/{model_name}.json') as jf:
    # loaded_model_json = jf.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(model_path, model_name + ".h5"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("Now inference...")
    y_pred_prob = model.predict_proba(test_x)
    return y_pred_prob


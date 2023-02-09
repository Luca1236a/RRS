#! /home/sooheang/anaconda3/bin/python

import argparse
import os
from os import mkdir
import pickle
import datetime

import numpy as np
import pandas as pd
import autogluon as ag

from autogluon import TabularPrediction as task
from autogluon.utils.tabular.metrics import roc_auc

# from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from DataClass import DataPath, VarSet
from mews import mews, mews_hr, mews_rr, mews_sbp, mews_bt

# from utils import make_RoC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb

from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser()
parser.add_argument("-d1", "--train_data", type=str, default="train_final.pickle")
parser.add_argument("-d2", "--test_data", type=str, default="test_final.pickle")
parser.add_argument("-d3", "--valid_data", type=str, default="valid_final.pickle")
parser.add_argument("-t", "--data_type", type=str, default="2D")
parser.add_argument("-s", "--scenario", type=str, default="sign")
parser.add_argument("-m", "--model", type=str, default="xgb")
parser.add_argument("-v", "--verbose", type=bool, default=True)

# scenario: str = 'full'
# model: str = 'xgb'
# verbose: bool = True
# data_type = "2D"
# train_data: str = 'train_final.pickle'
# test_data: str = 'test_final.pickle'
# valid_data: str = 'valid_final.pickle'


def main(args) -> None:

    train_data = args.train_data
    test_data = args.test_data
    valid_data = args.valid_data
    scenario = args.scenario
    model = args.model
    verbose = args.verbose
    data_type = args.data_type

    # load Path
    dp = DataPath()
    vs = VarSet()

    with open(os.path.join(dp.output_path, train_data), "rb") as f:
        train = pickle.load(f)

    with open(os.path.join(dp.output_path, test_data), "rb") as f:
        test = pickle.load(f)

    with open(os.path.join(dp.output_path, valid_data), "rb") as f:
        valid = pickle.load(f)

    df = pd.concat([train, test, valid])

    sign_list = [s for s in df.columns for i in vs.vital_grp if i in s]

    lab_a_list = [s for s in df.columns for i in vs.lab_grp_a if i in s]
    lab_b_list = [s for s in df.columns for i in vs.lab_grp_b if i in s]
    lab_c_list = [s for s in df.columns for i in vs.lab_grp_c if i in s]
    lab_d_list = [s for s in df.columns for i in vs.lab_grp_d if i in s]
    lab_e_list = [s for s in df.columns for i in vs.lab_grp_e if i in s]
    lab_f_list = [s for s in df.columns for i in vs.lab_grp_f if i in s]

    if data_type == "2D":
        meta_list = ["Age0", "Gender0"]
        time_list = []
    elif data_type == "3D":
        meta_list = ["Age", "Gender"]
        time_list = ["TS"]

    # define scenario
    if scenario == "full":
        var_list = [x for x in df.columns.tolist() if x not in ["target", "Patient"]]
    elif scenario == "sign":
        var_list = meta_list + time_list + sign_list
    elif scenario == "lab_A":
        var_list = meta_list + time_list + sign_list + lab_a_list
    elif scenario == "lab_B":
        var_list = meta_list + time_list + sign_list + lab_b_list
    elif scenario == "lab_C":
        var_list = meta_list + time_list + sign_list + lab_c_list
    elif scenario == "lab_D":
        var_list = meta_list + time_list + sign_list + lab_d_list
    elif scenario == "lab_E":
        var_list = meta_list + time_list + sign_list + lab_e_list
    elif scenario == "lab_F":
        var_list = meta_list + time_list + sign_list + lab_f_list
    elif scenario == "lab_AB":
        var_list = meta_list + time_list + sign_list + lab_a_list + lab_b_list
    elif scenario == "lab_ABC":
        var_list = meta_list + time_list + sign_list + lab_a_list + lab_b_list + lab_c_list
    elif scenario == "lab_ABCD":
        var_list = (
            meta_list + time_list + sign_list + lab_a_list + lab_b_list + lab_c_list + lab_d_list
        )

    if verbose:
        print("-" * 50)
        print("toal number of features: ", len(var_list))
        print("variables:", var_list)

    df[df == np.inf] = np.nan
    df = df[var_list + ["target", "Patient"]]

    # define fitted model
    if model in ["rf", "xgb", "lgb", "rnn"]:
        # df = df.dropna(axis = 0)

        id = df["Patient"].values
        X = df[var_list].fillna(df.median()).values
        y = df["target"].values

        if model == "rf":
            print("Fit Random Forest algorithm.")
            clf = RandomForestClassifier(n_estimators=500, n_jobs=30)
            param_grid = {
                "max_features": ["auto", "sqrt", "log2"],
                "max_depth": [10, 20, 30],
                "criterion": ["gini", "entropy"],
            }
        elif model == "xgb":
            print("Fit XGBoost algorithm")
            clf = xgb.XGBClassifier()
            param_grid = {
                "min_child_weight": [0.5, 1, 2, 4, 8, 12, 20],
                "max_depth": [9, 12, 15],
                "gamma": [0, 0.25, 0.5, 1],
                "n_estimators": [500],
                "nthread": [30],
                "objective": ["binary:logistic"],
            }

        elif model == "rnn":
            print("Fit RNN algorithm")
            clf = MLPClassifier()
            param_grid = {
                "activation": ["relu"],
                "solver": ["adam"],
                "alpha": [0.00001, 0.00005, 0.0001, 0.0005, 0.001],
                "learning_rate": ["adaptive"],
            }

        elif model == "lgb":
            print("Fit LightGBM Model")
            clf = LGBMClassifier(objective="binary")
            param_grid = {
                "boosting_type": ["gbdt"],
                "num_boost_round": [4000],
                "learning_rate": [0.001, 0.005, 0.01, 0.05],
                "metric": ["auc"],
            }

        fit = GridSearchCV(clf, param_grid=param_grid, scoring="f1")
        fit.fit(X, y)

        print("Model fitted")
        print("final params", fit.best_params_)
        print("best score", fit.best_score_)
        print("-" * 50)

    elif model == "automl":
        fit = AutoSklearn2Classifier(ml_memory_limit=None)
        fit.fit(X, y)

        print("Model fitted")
        print("best model: ", fit.show_models())

    elif model == "gluon":
        train = task.Dataset(df=df)
        label_column = "target"
        eval_metric = "f1"  #'roc_auc'
        output_directory = os.path.join(dp.model_path, "gluon", scenario)

        fit = task.fit(
            train_data=train.drop(["Patient"], axis=1),
            problem_type="binary",
            label=label_column,
            output_directory=output_directory,
            eval_metric=eval_metric,
            # presets = 'best_quality',
            verbosity=3,
            ngpus_per_trial=1,
        )

    elif model == "mews":
        mews_score = (
            df["HR"].apply(lambda x: mews_hr(x))
            + df["RR"].apply(lambda x: mews_rr(x))
            + df["BT"].apply(lambda x: mews_bt(x))
            + df["SBP"].apply(lambda x: mews_sbp(x))
        )
        fit = pd.DataFrame()
        fit["score"] = mews_score
        fit["Patient"] = df["Patient"]

    # cross_val_score(clf, X, y, cv = 5, scoring = 'recall_macro')

    # save the model
    if not os.path.isdir(dp.model_path):
        os.mkdir(dp.model_path)
    path = os.path.join(dp.model_path, model + "_" + scenario + ".pickle")

    print("saved path: ", path)
    with open(path, "wb") as f:
        pickle.dump(fit, f)

    return None


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


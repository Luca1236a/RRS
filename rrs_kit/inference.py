import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import autogluon as ag
from autogluon import TabularPrediction as task
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    auc,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    plot_precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from DataClass import DataPath, VarSet
from utils import plot_ROC_rev, plot_PR_curve
from mews import mews_hr, mews_rr, mews_sbp, mews_bt

# valid_file: str = 'valid_final.pickle'
valid_file: str = "valid_add_final.pickle"
# type: str = 'mews'
# type: str = 'gluon'
type: str = "xgb"
# type: str = 'rf'
# type: str = "lgb"
scenario: str = "lab_B"
model_file = type + "_" + scenario + ".pickle"


def infer_report(
    model_file: str = "xgb_lab_A.pickle", valid_file: str = "valid_add_final.pickle"
) -> pd.DataFrame:

    type = model_file.split("_", maxsplit=1)[0]
    scenario = model_file.split("_", maxsplit=1)[1].split(".")[0]

    # load Path
    dp = DataPath()
    vs = VarSet()

    if not type == "mews":
        with open(os.path.join(dp.model_path, model_file), "rb") as f:
            fit = pickle.load(f)

    with open(os.path.join(dp.valid_path, valid_file), "rb") as f:
        df = pickle.load(f)

    meta_list = ["Age0", "Gender0"]
    time_list = []
    sign_list = [s for s in df.columns for i in vs.vital_grp if i in s]

    lab_a_list = [s for s in df.columns for i in vs.lab_grp_a if i in s]
    lab_b_list = [s for s in df.columns for i in vs.lab_grp_b if i in s]
    lab_c_list = [s for s in df.columns for i in vs.lab_grp_c if i in s]
    lab_d_list = [s for s in df.columns for i in vs.lab_grp_d if i in s]
    lab_e_list = [s for s in df.columns for i in vs.lab_grp_e if i in s]
    lab_f_list = [s for s in df.columns for i in vs.lab_grp_f if i in s]

    if type == "mews":
        scenario == "mews"

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

    if type == "gluon":
        label_column = "target"
        id = df["Patient"].copy()
        df = df[var_list + [label_column]]
        df[df == np.inf] = np.nan
        # df['Patient'] = np.nan

        valid_data = task.Dataset(df)
        X_test = valid_data.drop([label_column], axis=1).values
        y_test = valid_data[label_column].values
        y_pred = fit.predict(valid_data)
        y_prob = fit.predict_proba(valid_data)
        # perf = fit.evaluate_predictions(y_test, y_pred)

        # predict per a patients
        threshold = 0.3
        y_pred[y_prob > threshold] = 1

        test = {"id": id, "true": y_test, "score": y_prob, "prediction": y_pred}

        df_test = pd.DataFrame(test)
        res = (
            df_test.groupby("id", group_keys=False)
            .apply(lambda x: x.tail(12))
            .groupby("id", as_index=False)
            .max()
        )

        # res = df_test.groupby('id', as_index = False).max()

        y_test = res["true"].tolist()
        y_pred = res["prediction"].tolist()
        y_prob = res["score"].round(4).tolist()

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print("predicted patients: \n", res["id"][res["prediction"] == 1].tolist())

        print([tn, fp, fn, tp])
        print("accuracy: ", accuracy_score(y_test, y_pred).round(4))
        print("f1: ", f1_score(y_test, y_pred).round(4))
        print("precision: ", precision_score(y_test, y_pred).round(4))
        print("recall: ", recall_score(y_test, y_pred).round(4))
        print("sensitivity: ", sensitivity.round(4))
        print("specificity: ", specificity.round(4))
        print("AUROC:", roc_auc_score(y_test, y_pred).round(4))

        plot_ROC_rev(y_test, y_pred)

        prs, rcs, _ = precision_recall_curve(y_test, y_pred)
        print("AUPRC:", auc(prs, rcs).round(4))
        plot_PR_curve(y_test, y_pred)

    elif type == "mews":
        # Calculate MEWS score
        y_prob = (
            df["HR0"].apply(lambda x: mews_hr(x))
            + df["RR0"].apply(lambda x: mews_rr(x))
            + df["BT0"].apply(lambda x: mews_bt(x))
            + df["SBP0"].apply(lambda x: mews_sbp(x))
        )

        id = df["Patient"].copy()
        y_pred = (y_prob >= 5).astype(int)

        y_test = df["target"].values

        test = {"id": id, "score": y_prob, "prediction": y_pred}

        df_test = pd.DataFrame(test)
        res = df_test.groupby("id", as_index=False).max()

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print("predicted patients: \n", res["id"][res["prediction"] == 1].tolist())

        print("accuracy: ", accuracy_score(y_test, y_pred).round(4))
        print("f1: ", f1_score(y_test, y_pred).round(4))
        print("precision: ", precision_score(y_test, y_pred).round(4))
        print("recall: ", recall_score(y_test, y_pred).round(4))
        print("sensitivity: ", sensitivity.round(4))
        print("specificity: ", specificity.round(4))
        print("AUROC:", roc_auc_score(y_test, y_pred).round(4))
        print("AUPRC:", average_precision_score(y_test, y_prob).round(4))

    else:
        X_test = df[var_list].fillna(df.median())
        y_test = df["target"].values
        y_pred = fit.predict(X_test)
        y_prob = fit.predict_proba(X_test)
        y_prob = np.array([x[0] for x in y_prob])

        # predict per a patients
        threshold = 0.3
        y_pred[y_prob > threshold] = 1

        id = df["Patient"].copy()
        test = {"id": id, "true": y_test, "score": y_prob, "prediction": y_pred}

        df_test = pd.DataFrame(test)
        res = (
            df_test.groupby("id", group_keys=False)
            .apply(lambda x: x.tail(12))
            .groupby("id", as_index=False)
            .max()
        )

        # res = df_test.groupby('id', as_index = False).max()

        y_test = res["true"].tolist()
        y_pred = res["prediction"].tolist()
        y_prob = res["score"].round(4).tolist()

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print("predicted patients: \n", res["id"][res["prediction"] == 1].tolist())

        print("accuracy: ", accuracy_score(y_test, y_pred).round(4))
        print("f1: ", f1_score(y_test, y_pred).round(4))
        print("precision: ", precision_score(y_test, y_pred).round(4))
        print("recall: ", recall_score(y_pred, y_test).round(4))
        print("sensitivity: ", sensitivity.round(4))
        print("specificity: ", specificity.round(4))
        print("AUROC:", roc_auc_score(y_test, y_pred))
        # print('AUPRC:', average_precision_score(y_test, y_score))

        plot_ROC_rev(y_test, y_pred)
        # plot_PR_curve(y_test, y_pred)
        plot_precision_recall_curve(fit, X_test, y_test)
        print(classification_report(y_test, y_pred, target_names=["0", "1"]))

    return res, id, y_prob, y_pred


if __name__ == "__main__":
    res_full, _, _, _ = infer_report("gluon_full.pickle")
    res_sign, _, _, _ = infer_report("gluon_sign.pickle")
    res_abcd, _, _, _ = infer_report("gluon_lab_ABCD.pickle")
    res_abc, _, _, _ = infer_report("gluon_lab_ABC.pickle")
    res_ab, _, _, _ = infer_report("gluon_lab_AB.pickle")
    res_a, _, _, _ = infer_report("gluon_lab_A.pickle")
    res_b, _, _, _ = infer_report("gluon_lab_B.pickle")
    res_c, _, _, _ = infer_report("gluon_lab_C.pickle")
    res_d, _, _, _ = infer_report("gluon_lab_D.pickle")
    res_e, _, _, _ = infer_report("gluon_lab_E.pickle")
    res_f, _, _, _ = infer_report("gluon_lab_F.pickle")
    res_mews, _, _, _ = infer_report("mews_full.pickle")

    res = res_full
    res = res.rename(columns={"score": "score_full", "prediction": "pred_full",})

    res["score_lab_abcd"] = res_abcd["score"]
    res["pred_lab_abcd"] = res_abcd["prediction"]

    res["score_lab_abc"] = res_abc["score"]
    res["pred_lab_abc"] = res_abc["prediction"]

    res["score_lab_ab"] = res_ab["score"]
    res["pred_lab_ab"] = res_ab["prediction"]

    res["score_lab_a"] = res_a["score"]
    res["pred_lab_a"] = res_a["prediction"]

    res["score_lab_b"] = res_b["score"]
    res["pred_lab_b"] = res_b["prediction"]

    res["score_lab_c"] = res_c["score"]
    res["pred_lab_c"] = res_c["prediction"]

    res["score_lab_d"] = res_d["score"]
    res["pred_lab_d"] = res_d["prediction"]

    res["score_lab_e"] = res_e["score"]
    res["pred_lab_e"] = res_e["prediction"]

    res["score_lab_f"] = res_f["score"]
    res["pred_lab_f"] = res_f["prediction"]

    res["score_mews"] = res_mews["score"]
    res["pred_mews"] = res_mews["prediction"]

    # res.to_csv('/data/datasets/res.csv')


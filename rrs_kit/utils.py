import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re
from inspect import signature

from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    auc,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

# from rrs_kit.DataClass import VarSet
from DataClass import VarSet
from datetime import timedelta


def filter_obs_times(m_time, e_time, day=1, hour=0):
    if m_time >= e_time - timedelta(days=day, hours=hour):
        return 1
    else:
        return 0


def get_var_list(columns: list, scenario: str) -> list:

    vs = VarSet()
    meta_list = ["Age0", "Gender0"]
    time_list = []

    mews_list = [s for s in columns for i in vs.vital_grp if i in s]
    sign_list = [s for s in columns for i in vs.vital_grp if i in s]

    lab_a_list = [s for s in columns for i in vs.lab_grp_a if i in s]
    lab_b_list = [s for s in columns for i in vs.lab_grp_b if i in s]
    lab_c_list = [s for s in columns for i in vs.lab_grp_c if i in s]
    lab_d_list = [s for s in columns for i in vs.lab_grp_d if i in s]
    lab_e_list = [s for s in columns for i in vs.lab_grp_e if i in s]
    lab_f_list = [s for s in columns for i in vs.lab_grp_f if i in s]

    if scenario == "full":
        var_list = [x for x in columns if x not in ["target", "Patient"]]
    elif scenario == "mews":
        var_list = mews_list
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

    return var_list


def fill_time(barcode_time, check_time, in_ward, before_detect, before_event):
    if str(barcode_time) == "nan":
        if check_time == "입원당일":
            return str(in_ward) + "1200"
        elif check_time == "detect발생1일전":
            return str(before_detect) + "1200"
        elif check_time == "event발생1일전":
            return str(before_event) + "1200"
    else:
        return str(barcode_time)


def fill_time_nl(barcode_time, check_time, in_ward, in_ward_5, discharge):
    if str(barcode_time) == "nan":
        if check_time == "입원당일":
            return str(in_ward) + "1200"
        elif check_time in ["입원5일후", "입원후4-10"]:
            return str(in_ward_5) + "1200"
        elif check_time == "퇴원":
            return str(discharge) + "1200"
    else:
        return str(barcode_time)


def drop_row(df: pd.DataFrame, drop_list) -> pd.DataFrame:
    df = df.drop(drop_list, axis=0)
    df = df.reset_index(drop=True)
    return df


def weird_time(df: pd.DataFrame):
    drop_index = []
    for i in range(len(df)):
        if len(df["측정시각"].iloc[i]) != 4:
            print(i, "th row is weird.", "-->", df["측정시각"].iloc[i])
            drop_index.append(i)
    return drop_index


def get_merge_data(df, df_bld):
    df_bld = get_resampled(df_bld, "patient_id")
    df_merged = df.merge(
        df_bld,
        left_on=["patient_id", "adjusted_time"],
        right_on=["patient_id", "adjusted_time"],
        how="left",
    )
    df_merged = get_resampled(df_merged, "patient_id", without_bfill=False)
    return df_merged


def get_resampled(df, index, time="adjusted_time", freq="1H", without_bfill=True):
    if without_bfill:
        df = df.groupby([index]).apply(lambda x: x.set_index(time).resample(freq).first().ffill())
        df = df.drop([index], axis=1)
        # df = df.dropna()
        df = df.reset_index(level=[1]).reset_index()
    else:
        df = df.groupby([index]).apply(
            lambda x: x.set_index(time).resample(freq).first().ffill().bfill()
        )
        df = df.drop([index], axis=1)
        # df = df.dropna()
        df = df.reset_index(level=[1]).reset_index()
    return df


def make_timestamp(df, index="일렬번호"):
    patient_list = np.unique(df[index])
    timestamp = []
    for i in patient_list:
        patient = df[df[index] == i]
        for t in range(len(patient)):
            timestamp.append(t + 1)
    return timestamp


def filter_sign(df, index, time="adjusted_time", freq="1H"):
    df["BT"].loc[df["BT"] > 43] = 43
    df["BT"].loc[df["BT"] < 35] = 35

    df["HR"].loc[df["HR"] > 300] = 300
    df["HR"].loc[df["HR"] < 30] = 30

    df["SaO2"].loc[df["SaO2"] > 100] = 100
    df["SaO2"].loc[df["SaO2"] < 65] = 65

    df["RR"].loc[df["RR"] > 40] = 40
    df["RR"].loc[df["RR"] < 6] = 6

    df["SBP"].loc[df["SBP"] > 210] = 210
    df["SBP"].loc[df["SBP"] < 60] = 60

    df = df.groupby([index]).apply(lambda x: x.set_index(time).resample(freq).first().ffill())
    df = df.drop([index], axis=1)
    df["SaO2"] = df["SaO2"].fillna(100)
    df = df.dropna()
    df = df.reset_index(level=[1]).reset_index()
    return df


def adjust_cbc(df: pd.DataFrame) -> pd.DataFrame:
    # CBC transformation
    df["WBC Count"] = np.where((4 <= df["WBC Count"]) & (df["WBC Count"] < 10.8), 1, 0)
    # df['Platelet']
    # df['Hgb']
    return df


def adjust_chem(df: pd.DataFrame) -> pd.DataFrame:

    # Chem
    # df['BUN']
    # df['Creatinin']
    df["Glucose"] = np.where(df["Glucose"] < 60, 0, np.where(df["Glucose"] < 100, 1, 2))
    df["Sodium"] = np.where(df["Sodium"] < 135, 0, np.where(df["Sodium"] < 145, 1, 2))
    df["Potassium"] = np.where(df["Potassium"] < 3.5, 0, np.where(df["Potassium"] < 5, 1, 2))
    df["Chloride"] = np.where(df["Chloride"] < 96, 0, np.where(df["Chloride"] < 108, 1, 2))

    return df


def get_target_revised(time, detection, event):
    try:
        condition = detection <= time <= event
    except:
        return 0
    if condition:
        return 1
    else:
        return 0


def get_target_df(df: pd.DataFrame, is_abn=True):
    if is_abn:
        df["target"] = df["event_time"] - df["adjusted_time"]
        df = df.drop(df[df["target"] <= pd.to_timedelta(0)].index)
        df["target"] = df.apply(
            lambda x: get_target_revised(x["adjusted_time"], x["detection_time"], x["event_time"]),
            axis=1,
        )
        return df
    else:
        df["target"] = 0
        return df


def get_datetime(df: pd.DataFrame, time_var: str = "measurement_time") -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df[time_var].astype(str))
    hour_delta = pd.to_timedelta(df["hour"].astype("int"), unit="h")
    min_delta = pd.to_timedelta(df["minute"].astype("int"), unit="m")
    df["datetime"] += hour_delta + min_delta
    return df


def get_datetime_event(df, target_column, date_column, hour_column):
    df[target_column] = pd.to_datetime(df[date_column])
    hour_delta = pd.to_timedelta(df[hour_column].astype("int"), unit="h")
    df[target_column] += hour_delta
    return df


def abn_nl_concat(abn_data, nl_data):
    abn_data["Patient"] = max(nl_data["Patient"]) + abn_data["Patient"]
    concat_data = pd.concat([nl_data, abn_data], axis=0, sort=False)
    return concat_data


def make_sequence_data(
    df, window_len, var_list, index="patient_id", is_abn="is_abn", target_list="target"
):

    # return_df = pd.DataFrame(columns=[index, is_abn, 'sequence', target_list])
    return_df = pd.DataFrame(
        columns=[
            index,
            is_abn,
            "sequence",
            target_list,
            "measurement_time",
            "detection_time",
            "event_time",
        ]
    )
    patient_list = df[index].unique()

    print("총 환자수:", len(patient_list))
    print("Window 크기:", window_len)
    print("-" * 20, "Making Data", "-" * 20)

    row_num = 1
    for i in patient_list:
        if row_num % 100 == 0:
            print("=" * int(row_num / 100) + ">", str(row_num) + "번 째 환자")

        target = []
        sequence = []
        patient = df[df[index] == i]
        patient = patient.reset_index(drop=True)
        id_list = []
        is_abn_list = []
        measurement_time = []
        event_time = []
        detection_time = []

        for j in range(len(patient) - window_len):
            id_list.append(patient[index].iloc[j])
            is_abn_list.append(patient[is_abn].iloc[j])
            target.append(patient[target_list].iloc[j + window_len - 1])
            sequence.append((patient[var_list].iloc[j : j + window_len]).values)
            measurement_time.append(patient["measurement_time"].iloc[j])
            detection_time.append(patient["detection_time"].iloc[j])
            event_time.append(patient["event_time"].iloc[j])

        row_num += 1
        new_df = {
            "Patient": id_list,
            "is_abn": is_abn_list,
            "measurement_time": measurement_time,
            "detection_time": detection_time,
            "event_time": event_time,
            "sequence": sequence,
            "target": target,
        }
        new_df = pd.DataFrame(new_df)
        return_df = pd.concat([return_df, new_df])
    print("-" * 20, "Done", "-" * 20)
    return return_df


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


def make_multi_column(col, window=8):
    idx = []
    column_name = [col]
    for i in column_name:
        for n in range(window):
            idx.append(i + str(n - (window - 1)))
    return idx


def make_multi_column_2(col, window=8):
    idx = []
    column_name = [col]
    for i in column_name:
        for n in range(window - 1):
            idx.append(i + str(n - (window - 1)))
    return idx


def make_RoC(df, col_list, col_zero):
    for i in col_list:
        try:
            df["RoC_" + str(i)] = (df[col_zero] - df[i]) / df[i]
        except:
            df["RoC_" + str(i)] = df[col_zero]
    return df


def make_statistic(df, col_list, col_name):
    df["mean_" + str(col_name)] = np.mean(df[col_list], axis=1)
    df["std_" + str(col_name)] = np.std(df[col_list], axis=1)
    df["max_" + str(col_name)] = np.max(df[col_list], axis=1)
    df["min_" + str(col_name)] = np.min(df[col_list], axis=1)
    return df


def make_2d_time(df: pd.DataFrame, time_var_list: list, window_len: int = 8) -> pd.DataFrame:
    for var in time_var_list:
        col_list = make_multi_column(var, window_len)
        df = make_statistic(df, col_list, var)
        col_list = make_multi_column_2(var, window_len)
        df = make_RoC(df, col_list, var + "0")
    return df


def make_2d_data(
    df_whole, var_list, output_path, output_file="train_final.csv", window_len=8
) -> pd.DataFrame:

    X = np.stack(df_whole.sequence.values, axis=0)
    y = np.stack(df_whole.target.values, axis=0).reshape(-1, 1)
    id = np.stack(df_whole.Patient.values, axis=0).reshape(-1, 1)
    time1 = np.stack(df_whole.measurement_time.values, axis=0).reshape(-1, 1)
    time2 = np.stack(df_whole.detection_time.values, axis=0).reshape(-1, 1)
    time3 = np.stack(df_whole.event_time.values, axis=0).reshape(-1, 1)

    column_name_multi = []
    for n in range(window_len):
        for i in var_list:
            column_name_multi.append(i + str(n - (window_len - 1)))
    print("column_name_multi: ", column_name_multi)

    X = X.reshape(len(X), len(var_list) * window_len)

    X_2d = pd.DataFrame(data=X, columns=column_name_multi)
    y_2d = pd.DataFrame(data=y, columns=["target"])
    id_2d = pd.DataFrame(data=id, columns=["Patient"])
    time1_2d = pd.DataFrame(data=time1, columns=["measurement_time"])
    time2_2d = pd.DataFrame(data=time2, columns=["detection_time"])
    time3_2d = pd.DataFrame(data=time3, columns=["event_time"])

    drop_list = ["Gender", "Age", "TS"]
    drop_name_multi = []
    for n in range(window_len - 1):
        for i in drop_list:
            drop_name_multi.append(i + str(n - (window_len - 1)))
    print("drop list: ", drop_name_multi)

    X_2d = X_2d.drop(drop_name_multi, axis=1)

    exp_list = ["Gender", "Age", "TS", "is_abn", "measurement_time", "event_time", "detection_time"]
    time_series = [v for v in var_list if v not in exp_list]

    print("make 2D time data")
    X_2d = make_2d_time(X_2d, time_series)

    ds = pd.concat([X_2d, y_2d, id_2d, time1_2d, time2_2d, time3_2d], axis=1)
    ds.to_csv(os.path.join(output_path, output_file), index=False)

    path = os.path.join(output_path, os.path.splitext(output_file)[0] + ".pickle")
    print("Saved pickle: ", path)
    with open(path, "wb") as f:
        pickle.dump(ds, f)
    print("Done.")
    return ds


def plot_ROC_rev(test_labels, test_predictions):
    fpr, tpr, thr = roc_curve(test_labels, test_predictions, pos_label=1)
    aucs = "%.5f" % auc(fpr, tpr)
    title = "ROC Curve, AUC = " + str(aucs)
    # Optimal threshold

    # tnr > 0.95
    tnr_goal_95 = np.where(1 - fpr > 0.95)
    tnr_goal_95 = tnr_goal_95[0]
    # maximum tpr
    opt_95 = tnr_goal_95[np.argmax(tpr[tnr_goal_95])]

    # tnr > 0.99
    tnr_goal_99 = np.where(1 - fpr > 0.99)
    tnr_goal_99 = tnr_goal_99[0]
    # maximum tpr
    opt_99 = tnr_goal_99[np.argmax(tpr[tnr_goal_99])]

    # tnr > 0.90
    tnr_goal_90 = np.where(1 - fpr > 0.90)
    tnr_goal_90 = tnr_goal_90[0]
    # maximum tpr
    opt_90 = tnr_goal_90[np.argmax(tpr[tnr_goal_90])]

    # tnr > 0.85
    tnr_goal_85 = np.where(1 - fpr > 0.85)
    tnr_goal_85 = tnr_goal_85[0]
    # maximum tpr
    opt_85 = tnr_goal_85[np.argmax(tpr[tnr_goal_85])]

    with plt.style.context(("ggplot")):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, "#1c3768", label="ROC curve")
        ax.plot(fpr[opt_99], tpr[opt_99], "ro", label="TNR>.99")
        ax.plot(fpr[opt_95], tpr[opt_95], "go", label="TNR>.95")
        ax.plot(fpr[opt_90], tpr[opt_90], "bo", label="TNR>.90")
        ax.plot(fpr[opt_85], tpr[opt_85], "yo", label="TNR>.85")
        ax.plot([0, 1], [0, 1], "k--", label="Baseline")
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.legend(loc="lower right")
        plt.title(title)
    plt.show()
    print(f"ROC AUC Score : {roc_auc_score(test_labels, test_predictions)}")
    print(f"TNR>0.99 Threshold : {thr[opt_99]}, tpr : {tpr[opt_99]}, fpr : {fpr[opt_99]}")
    print(f"TNR>0.95 Threshold : {thr[opt_95]}, tpr : {tpr[opt_95]}, fpr : {fpr[opt_95]}")
    print(f"TNR>0.90 Threshold : {thr[opt_90]}, tpr : {tpr[opt_90]}, fpr : {fpr[opt_90]}")
    print(f"TNR>0.85 Threshold : {thr[opt_85]}, tpr : {tpr[opt_85]}, fpr : {fpr[opt_85]}")


def plot_ROC(test_labels, test_predictions):
    fpr, tpr, thr = roc_curve(test_labels, test_predictions, pos_label=1)
    aucs = "%.5f" % auc(fpr, tpr)
    title = "ROC Curve, AUC = " + str(aucs)
    # Optimal threshold

    # tnr > 0.95
    tnr_goal_idx = np.where(1 - fpr > 0.95)
    tnr_goal_idx = tnr_goal_idx[0]

    # maximum tpr
    opt_idx = tnr_goal_idx[np.argmax(tpr[tnr_goal_idx])]

    # max tpr+(1-fpr)
    conv_opt_idx = np.argmax(tpr + (1 - fpr))

    with plt.style.context(("ggplot")):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, "#1c3768", label="ROC curve")
        ax.plot(fpr[opt_idx], tpr[opt_idx], "ro", label="MAX TPR")
        ax.plot(fpr[conv_opt_idx], tpr[conv_opt_idx], "bo", label="MAX TPR+(1-FPR)")
        ax.plot([0, 1], [0, 1], "k--", label="Baseline")
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.legend(loc="lower right")
        plt.title(title)
    plt.show()
    print(f"ROC AUC Score : {roc_auc_score(test_labels, test_predictions)}")
    print(
        f"Conventional Threshold : {thr[conv_opt_idx]}, tpr : {tpr[conv_opt_idx]}, fpr : {fpr[conv_opt_idx]}"
    )
    print(f"TNR>0.95 Threshold : {thr[opt_idx]}, tpr : {tpr[opt_idx]}, fpr : {fpr[opt_idx]}")


from matplotlib import pyplot


def plot_PR_curve(test_labels, test_predictions):

    average_precision = average_precision_score(test_labels, test_predictions)
    precision, recall, _ = precision_recall_curve(test_labels, test_predictions)

    # In matplotlib < 1.5, plt.`fill_`between does not have a 'step' argument
    aucs = "%.5f" % auc(precision, recall)

    pyplot.plot(recall, precision, marker=".", label="RNN-XGB")
    # axis labels
    pyplot.xlabel("Recall")
    pyplot.ylabel("Precision")
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    title = "PR Curve, AU PRC = " + str(aucs)
    with plt.style.context(("ggplot")):
        fig, ax = plt.subplots()
        ax.plot(recall, precision, "#1c3768", label="PR curve")
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower right")
        plt.title(title)
    plt.show()


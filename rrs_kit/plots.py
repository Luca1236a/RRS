import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import (
  confusion_matrix,
  precision_recall_curve,
  average_precision_score
)

def plot_PR_curve(test_labels, test_predictions):
  average_precision = average_precision_score(test_labels, test_predictions)
  precision, recall, _ = precision_recall_curve(test_labels, test_predictions)
  # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
  aucs = "%.5f" % auc(precision, recall, reorder=True)
  step_kwargs = ({'step': 'post'}
                  if 'step' in signature(plt.fill_between).parameters
                  else {})
  plt.step(recall, precision, color='b', alpha=0.2, where='post')
  plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('2-class Precision-Recall curve: PRAUC=' + aucs)
  plt.show()
  print('PR AUC Score:', aucs)
  
def plot_ROC_rev(test_labels, test_predictions):
  fpr, tpr, thr = roc_curve(
    test_labels, test_predictions, pos_label=1) 
  aucs = "%.5f" % auc(fpr, tpr)
  title = 'ROC Curve, AUC = '+str(aucs)
  # Optimal threshold

  # tnr > 0.95
  tnr_goal_95 = np.where(1-fpr>0.95)
  tnr_goal_95=tnr_goal_95[0]
  # maximum tpr
  opt_95 = tnr_goal_95[np.argmax(tpr[tnr_goal_95])]
  
  # tnr > 0.99
  tnr_goal_99 = np.where(1-fpr>0.99)
  tnr_goal_99=tnr_goal_99[0]
  # maximum tpr
  opt_99 = tnr_goal_99[np.argmax(tpr[tnr_goal_99])]
  
  # tnr > 0.90
  tnr_goal_90 = np.where(1-fpr>0.90)
  tnr_goal_90=tnr_goal_90[0]
  # maximum tpr
  opt_90 = tnr_goal_90[np.argmax(tpr[tnr_goal_90])]
  
  # tnr > 0.85
  tnr_goal_85 = np.where(1-fpr>0.85)
  tnr_goal_85=tnr_goal_85[0]
  # maximum tpr
  opt_85 = tnr_goal_85[np.argmax(tpr[tnr_goal_85])]
  
  with plt.style.context(('ggplot')):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, "#1c3768", label='ROC curve')
    ax.plot(fpr[opt_99], tpr[opt_99], 'ro', label = 'TNR>.99')
    ax.plot(fpr[opt_95], tpr[opt_95], 'go', label = 'TNR>.95')
    ax.plot(fpr[opt_90], tpr[opt_90], 'bo', label = 'TNR>.90')
    ax.plot(fpr[opt_85], tpr[opt_85], 'yo', label = 'TNR>.85')
    ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.title(title)
  plt.show()

  print(f"ROC AUC Score : {roc_auc_score(test_labels, test_predictions)}")
  print(f'TNR>0.99 Threshold : {thr[opt_99]}, tpr : {tpr[opt_99]}, fpr : {fpr[opt_99]}')
  print(f'TNR>0.95 Threshold : {thr[opt_95]}, tpr : {tpr[opt_95]}, fpr : {fpr[opt_95]}')
  print(f'TNR>0.90 Threshold : {thr[opt_90]}, tpr : {tpr[opt_90]}, fpr : {fpr[opt_90]}')
  print(f'TNR>0.85 Threshold : {thr[opt_85]}, tpr : {tpr[opt_85]}, fpr : {fpr[opt_85]}')


def plot_ROC(test_labels, test_predictions):
  fpr, tpr, thr = roc_curve(
      test_labels, test_predictions, pos_label=1) 
  aucs = "%.5f" % auc(fpr, tpr)
  title = 'ROC Curve, AUC = '+str(aucs)
  # Optimal threshold

  # tnr > 0.95
  tnr_goal_idx = np.where(1-fpr>0.95)
  tnr_goal_idx=tnr_goal_idx[0]

  # maximum tpr
  opt_idx = tnr_goal_idx[np.argmax(tpr[tnr_goal_idx])]

  # max tpr+(1-fpr)
  conv_opt_idx = np.argmax(tpr + (1-fpr))
  
  with plt.style.context(('ggplot')):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, "#1c3768", label='ROC curve')
    ax.plot(fpr[opt_idx], tpr[opt_idx], 'ro', label = 'MAX TPR')
    ax.plot(fpr[conv_opt_idx], tpr[conv_opt_idx], 'bo', label ='MAX TPR+(1-FPR)')
    ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.title(title)

  plt.show()
  
  print(f"ROC AUC Score : {roc_auc_score(test_labels, test_predictions)}")
  print(f'Conventional Threshold : {thr[conv_opt_idx]}, tpr : {tpr[conv_opt_idx]}, fpr : {fpr[conv_opt_idx]}')
  print(f'TNR>0.95 Threshold : {thr[opt_idx]}, tpr : {tpr[opt_idx]}, fpr : {fpr[opt_idx]}')
  
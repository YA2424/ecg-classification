from sklearn import metrics as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_metrics(y_test,y_pred):
    metrics = {}
    metrics['accuracy'] = m.accuracy_score(y_test,y_pred)
    metrics['precision'] = m.precision_score(y_test, y_pred,average='weighted')
    metrics['recall'] = m.recall_score(y_test, y_pred,average='weighted')
    metrics['f1'] = m.f1_score(y_test, y_pred,average='weighted')
    metrics['balanced_accuracy']=m.balanced_accuracy_score(y_test, y_pred)
    return metrics

def get_classification_report(y_test, y_pred):
    report=pd.DataFrame(m.classification_report(y_test, y_pred, output_dict=True)).T
    report.to_csv('src/reports/classification_report.csv')

def get_features_importances(features,importances):
    df=pd.DataFrame({'features':features, 'importances':importances})
    df=df.sort_values('importances', ascending=False)
    df=df.reset_index(drop=True)
    df.to_csv('src/reports/feature_importances.csv',index=False)
    return df

def get_feature_importances_plot(df):
    df=df.sort_values('importances', ascending=True)
    figure = plt.figure(figsize=(10,10))
    plt.barh(df['features'], df['importances'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    return figure


def get_confusion_matrix_plot(y_test,y_pred):
    confusion_matrix=pd.DataFrame(m.confusion_matrix(y_test, y_pred))
    sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.savefig('src/reports/confusion_matrix.png')
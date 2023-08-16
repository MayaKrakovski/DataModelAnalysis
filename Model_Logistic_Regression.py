import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, \
    roc_curve, auc
import pickle
import math


def print_summary(logit_model, t):
    plt.figure()
    plt.title(t)
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(logit_model.summary2()), {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def logistic_by_ex(df, features, X_train, X_test, y_train, y_test, return_models=False):
    exercises = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows']
    features_list = []
    models_dict = {}
    for e in exercises:
        df_ex = df[df['Exercise'] == e]
        ex_train_indx = list(set(df_ex.index).intersection(X_train.index))
        df_ex_train = df.iloc[list(set(df_ex.index).intersection(X_train.index))]
        df_ex_test = df.iloc[list(set(df_ex.index).intersection(X_test.index))]
        X_train_ex = df_ex_train[features]
        y_train_ex = df_ex_train["label"]
        X_test_ex = df_ex_test[features]
        y_test_ex = df_ex_test["label"]

        logit_model = sm.Logit(y_train_ex, X_train_ex).fit_regularized(method='l1', alpha=1)
        logit_model.summary2()
        print_summary(logit_model, e)
        result_df = model_evaluation(logit_model, X_test_ex, y_test_ex)
        # predictions = logit_model.predict(X_test_ex)
        # predictions = list(map(round, predictions))
        # print(classification_report(y_test_ex, predictions))
        # print(confusion_matrix(y_test_ex, predictions))
        models_dict[e] = logit_model

        var = logit_model.params
        var = list(var[var != 0].index)
        features_list.append(var)
    if return_models:
        return models_dict
    else:
        return features_list


def logistic(X_train, X_test, y_train, y_test, alpha, title):
    logit_model = sm.Logit(y_train, X_train).fit_regularized(method='l1', alpha=alpha)
    print_summary(logit_model, alpha)
    model_evaluation(logit_model, X_test, y_test)
    return logit_model


def model_evaluation(model, X_test, y_test):
    predictions = model.predict(X_test)
    predictions = list(map(round, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    # PR-RECALL auc
    roc_auc = roc_auc_score(y_test, predictions)
    precision, recall, _ = precision_recall_curve(y_test, predictions)
    pr_auc = auc(recall, precision)
    print(f"ROC AUC: {roc_auc}, Precision-Recall AUC: {pr_auc}")

    result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    return result_df


def models():
    # read Data and filter to maya source only
    file_name = 'allfeatures_nonscaled_label.csv'  # File containing all data
    file_name = 'allfeatures_scaled_label.csv'  # File containing all data
    file_name = 'allfeatures_raw_scaled_label.csv'  # File containing all data
    df = pd.read_csv(f'CSV/features/{file_name}')

    df_maya = df[df['Source'] == 'maya']
    features = df_maya.columns[6:]
    features = features[1:-2]
    X = df_maya[features]
    y = df_maya["label"]
    # split 80 train 20 test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model 1
    model1 = logistic(X_train, X_test, y_train, y_test, 1, 'Model 1 - All Features with L1')
    pickle.dump(model1, open('model1r.sav', 'wb'))

    # selected features for model 2 and 3
    selected_features = logistic_by_ex(df, features, X_train, X_test, y_train, y_test)
    flat_list = [item for sublist in selected_features for item in sublist]
    flat_set = set(flat_list)
    X_train_sel = X_train[flat_set]
    X_test_sel = X_test[flat_set]

    # model 2
    model2 = logistic(X_train_sel, X_test_sel, y_train, y_test, 1, 'Model 2 - Selected Features with L1')
    pickle.dump(model2, open('model2r.sav', 'wb'))
    # model 3
    model3 = logistic(X_train_sel, X_test_sel, y_train, y_test, 0, 'Model 3 - Selected Features')
    pickle.dump(model3, open('model3r.sav', 'wb'))


def test_data(model):
    file_name = 'allfeatures_raw_scaled_label.csv' # File containing all data
    df = pd.read_csv(f'CSV/features/{file_name}')
    features = model.model.data.xnames

    data_sources = ['naama', 'naama_pilot', 'val1']
    for ds in data_sources:
        print(f"-------------------{ds}------------------")
        df_test = df[df['Source'] == ds]
        X_test = df_test[features]
        y_test = df_test["label"]

        result_df = model_evaluation(model, X_test, y_test)
        df_test['prediction'] = result_df.loc[result_df.index, 'Predicted']
        result_df['Participant'] = df.loc[df_test.index, 'Participant']
        result_df['hand'] = df.loc[df_test.index, 'hand']
        result_df['Exercise'] = df.loc[df_test.index, 'Exercise']

        # exercise_counts = result_df.groupby(['Exercise', 'Result']).size().reset_index(name='Count').to_csv("aaaaa.csv")i

        def label_record(row):
            if row['Actual'] == 1 and row['Predicted'] == 1:
                return 'TP'
            elif row['Actual'] == 0 and row['Predicted'] == 0:
                return 'TN'
            elif row['Actual'] == 0 and row['Predicted'] == 1:
                return 'FP'
            elif row['Actual'] == 1 and row['Predicted'] == 0:
                return 'FN'

        # Apply the function to create a new column 'Result' with TP, TN, FP, or FN labels
        result_df['Result'] = result_df.apply(label_record, axis=1)
        result_df.groupby(['Exercise', 'Result']).size().reset_index(name='Count')

    return result_df


    # # labels for val1 by plots and not instructions
    # df_labels = pd.read_csv(r"C:\Users\mayak\PycharmProjects\DataAnalysis\CSV\Raw Data\val1_raw_data_scaled_labeled.csv")
    #
    # join_columns = ['Participant', 'Exercise', 'hand']
    #
    # combined_df = pd.merge(df, df_labels, on=join_columns, how='inner')
    # y_test = combined_df["Label by plot"]


def test_model_per_ex(df, models_dict):

    data_sources = ['naama', 'naama_pilot', 'val1']
    exercises = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows']

    models_dict = logistic_by_ex(df, features, X_train, X_test, y_train, y_test, True)

    for ds in data_sources:
        print(f'----------{ds}-----------')
        df_test = df[df['Source'] == ds]
        for e in exercises:
            print(f'-------{e}-------')
            ex_model = models_dict[e]
            features = ex_model.model.data.xnames
            df_ex = df_test[df_test['Exercise'] == e]
            X_test = df_ex[features]
            y_test = df_ex["label"]

            result_df = model_evaluation(ex_model, X_test, y_test)


def plot_signals(result_df):
    path = r"C:\Users\mayak\PycharmProjects\DataAnalysis\CSV\Raw Data\all_raw_data_scaled.csv"
    df = pd.read_csv(path)

    join_columns = ['Participant', 'Exercise', 'hand']
    combined_df = pd.merge(df, result_df, on=join_columns, how='inner')
    signalcols = combined_df.columns[combined_df.columns.str.startswith('Unnamed')][1:]

    for e in combined_df['Exercise'].unique():
        fig = plt.figure()
        df_ex = combined_df[combined_df["Exercise"] == e]
        df_ex_FN = df_ex[df_ex['Result']=='FN']
        amount = len(df_ex_FN.index)
        fig_count = 0
        for i in range(0, len(df_ex_FN.index)):
            fig_count += 1
            temp = df_ex_FN.iloc[i,:][signalcols].dropna().to_numpy()
            ax = fig.add_subplot(4, math.ceil(amount/4), fig_count)
            ax.plot(temp)
        fig.suptitle(f"{e}: FN")

if __name__ == '__main__':
    # models()

    # pickle.dump(logit_model, open('model1.sav', 'wb'))

    for m in ['model1r']:
        model = pickle.load(open(f'{m}.sav', 'rb'))

        test_data(model)


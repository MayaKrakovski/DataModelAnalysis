import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle


def print_summary(logit_model, t):
    plt.figure()
    plt.title(t)
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(logit_model.summary2()), {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def logistic_by_ex(df_maya, features, X_train, X_test, y_train, y_test):
    exercises = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows']
    features_list = []
    for e in exercises:
        df_ex = df_maya[df_maya['Exercise'] == e]
        ex_train_indx = list(set(df_ex.index).intersection(X_train.index))
        df_ex_train = df_maya.iloc[list(set(df_ex.index).intersection(X_train.index))]
        df_ex_test = df_maya.iloc[list(set(df_ex.index).intersection(X_test.index))]
        X_train_ex = df_ex_train[features]
        y_train_ex = df_ex_train["label"]
        X_test_ex = df_ex_test[features]
        y_test_ex = df_ex_test["label"]

        logit_model = sm.Logit(y_train_ex, X_train_ex).fit_regularized(method='l1', alpha=1)
        logit_model.summary2()
        print_summary(logit_model, e)
        predictions = logit_model.predict(X_test_ex)
        predictions = list(map(round, predictions))
        print(classification_report(y_test_ex, predictions))
        print(confusion_matrix(y_test_ex, predictions))

        var = logit_model.params
        var = list(var[var != 0].index)
        features_list.append(var)

    return features_list


def logistic(X_train, X_test, y_train, y_test, alpha, title):
    logit_model = sm.Logit(y_train, X_train).fit_regularized(method='l1', alpha=alpha)
    print_summary(logit_model, title)
    model_evaluation(logit_model, X_test, y_test)
    return model


def model_evaluation(model, X_test, y_test):
    predictions = model.predict(X_test)
    predictions = list(map(round, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


def models():
    # read Data and filter to maya source only
    file_name = 'allfeaturesbyhandscaled_label.csv' # File containing all data
    df = pd.read_csv(f'CSV/features/{file_name}')

    df_maya = df[df['Source'] == 'maya']
    features = df_maya.columns[5:]
    X = df_maya[features]
    y = df_maya["label"]
    # split 80 train 20 test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model 1
    model1 = logistic(X_train, X_test, y_train, y_test, 1, 'Model 1 - All Features with L1')
    pickle.dump(model1, open('model1.sav', 'wb'))

    # selected features for model 2 and 3
    selected_features = logistic_by_ex(df_maya, features, X_train, X_test, y_train, y_test)
    flat_list = [item for sublist in selected_features for item in sublist]
    flat_set = set(flat_list)
    X_train_sel = X_train[flat_set]
    X_test_sel = X_test[flat_set]

    # model 2
    model2 = logistic(X_train_sel, X_test_sel, y_train, y_test, 1, 'Model 2 - Selected Features with L1')
    pickle.dump(model2, open('model2.sav', 'wb'))
    # model 3
    model3 = logistic(X_train_sel, X_test_sel, y_train, y_test, 0, 'Model 3 - Selected Features')
    pickle.dump(model3, open('model3.sav', 'wb'))


def test_data(model):
    file_name = 'allfeaturesbyhandscaled_label.csv' # File containing all data
    df = pd.read_csv(f'CSV/features/{file_name}')
    features = model.model.data.xnames

    data_sources = ['naama', 'naama_pilot']
    for ds in data_sources:
        print(f"-------------------{ds}------------------")
        df_test = df[df['Source'] == ds]
        X_test = df_test[features]
        y_test = df_test["label"]

        model_evaluation(model, X_test, y_test)


if __name__ == '__main__':
    # models()

    # pickle.dump(logit_model, open('model1.sav', 'wb'))

    for m in ['model1', 'model2', 'model3']:
        model = pickle.load(open(f'{m}.sav', 'rb'))

        test_data(model)

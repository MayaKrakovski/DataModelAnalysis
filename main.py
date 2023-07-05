import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def start(df, exercise='all'):
    """
    :param exercise: 'all' - for all three exercise, exercise name - for specific exercise/s
    :param df: data frame
    """
    # filter df if required by exercise
    if exercise != 'all':
        df = df[df["exercise"].isin(exercise)]

    # split features and labels
    features = df.columns[3:-1]
    X = df[features]
    y = df["label"]

    return regression(X, y, df) # Calculate regression model


def regression(X, y, df, summary=True):
    # model = LogisticRegression(penalty='l1', solver='liblinear').fit(X, y)
    # print(model.coef_)
    # model.summary()

    # Logistic regression model with L1 regularization, alpha = penalty weight
    logit_model = sm.Logit(y, X).fit_regularized(method='l1', alpha=1)
    var = logit_model.params

    # get the selected features by the regression model
    var = list(var[var != 0].index)
    X = X[var]

    if summary:
        print_summary(logit_model)

    return var


def print_summary(logit_model):
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(logit_model.summary2()), {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # read data - features already scaled
    file_name = '3ex_scale_data_labeled.csv' # File containing all data
    data = pd.read_csv(file_name)

    exercise_names = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows']
    features_list = []
    features_list.append(start(data))
    for e in exercise_names:
        print(e)
        features_list.append(start(data, [e]))

    # list of selected features for multiple exercises
    flat_list = [item for sublist in features_list for item in sublist]
    flat_set = set(flat_list)

    # regression with only the selected features:
    features = data.columns[3:-1]
    X = data[flat_set]
    y = data["label"]
    logit_model = sm.Logit(y, X).fit_regularized(method='l1', alpha=1)
    logit_model.summary2()
    print_summary(logit_model)

    # check with train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logit_model = sm.Logit(y_train, X_train).fit_regularized(method='l1', alpha=0)
    logit_model.summary2()
    print_summary(logit_model)
    predictions = logit_model.predict(X_test)
    predictions = list(map(round, predictions))
    print(classification_report(y_test, predictions))
    confusion_matrix(y_test, predictions)



# list of intersection features only
    ab = list(set(features_list[0]).intersection(features_list[1]).intersection(features_list[2]))

    # regression with only the intersection features:
    X = data[ab]
    y = data["label"]
    logit_model = sm.Logit(y, X).fit_regularized(method='l1', alpha=1)
    logit_model.summary2()
    print_summary(logit_model)

    # list of intersection features only for a+b exercises
    ab = list(set(features_list[0]).intersection(features_list[1]))

    # regression with only the intersection features:
    X = data[data["exercise"].isin(exercise_names[0:2])][ab]
    y = data[data["exercise"].isin(exercise_names[0:2])]["label"]
    logit_model = sm.Logit(y, X).fit()
    logit_model.summary2()
    print_summary(logit_model)

    # list of intersection features only for a+b exercises
    ab = list(set(features_list[1]).intersection(features_list[2]))

    # regression with only the intersection features:
    X = data[data["exercise"].isin(exercise_names[1:])][ab]
    y = data[data["exercise"].isin(exercise_names[1:])]["label"]
    logit_model = sm.Logit(y, X).fit()
    logit_model.summary2()
    print_summary(logit_model)

    # list of intersection features only for a+b exercises
    ab = list(set(features_list[0]).intersection(features_list[2]))

    # regression with only the intersection features:
    X = data[data["exercise"].isin([exercise_names[0],exercise_names[2]])][ab]
    y = data[data["exercise"].isin([exercise_names[0],exercise_names[2]])]["label"]
    logit_model = sm.Logit(y, X).fit()
    logit_model.summary2()
    print_summary(logit_model)


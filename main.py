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
    file_name = 'CSV/features/old/3ex_scale_data_labeled.csv'  # File containing all data
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




data = [
    ["Trained on older adults 1", "Non Scaled", 0.87, 0.84],
    ["Trained on older adults 1", "Feature Scaled", 0.89, 0.62],
    ["Trained on older adults 1", "Raw Scaled", 0.84, 0.82],
    ["Trained on Validation", "Non Scaled", 0.73, 0.96],
    ["Trained on Validation", "Feature Scaled", 0.83, 0.98],
    ["Trained on Validation", "Feature Scaled by Validation", 0.59, 0.94],
    ["Trained on Validation", "Raw Scaled", 0.68, 0.96],
    ["Train on Both:", "Non Scaled", 0.8, 0.97],
    ["Train on Both:", "Feature Scaled", 0.96, 0.97],
    ["Train on Both:", "Feature Scaled by Validation", 0.93, 0.95],
    ["Train on Both:", "Raw Scaled", 0.82, 0.93]
]

columns = ["Training Scenario", "Scaling Type", "Older Adults 1", "Validation"]
df = pd.DataFrame(data, columns=columns)

# Create a dictionary to map training scenarios to colors
sns.set_palette("husl", 3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot for Older Adults 1
handles = []
labels = []
for training_scenario, group in df.groupby("Training Scenario"):
    line = ax1.plot(
        group["Scaling Type"],
        group["Older Adults 1"],
        marker='o',
        linestyle='-',
        label=training_scenario
    )[0]
    handles.append(line)
    labels.append(training_scenario)
ax1.set_xlabel("Scaling Type")
ax1.set_ylabel("Accuracy")
ax1.set_title("Older Adults 1 vs Scaling Type")
ax1.set_xticks(range(len(df["Scaling Type"].unique())))  # Set tick positions
ax1.set_xticklabels(['Non Scaled', 'Feature Scaled by Older Adults 1',
       'Feature Scaled by Validation', 'Raw Scaled'], rotation=45)  # Set tick labels
ax1.set_ylim(0, 1)
ax1.grid(True)

# Plot for Validation
for training_scenario, group in df.groupby("Training Scenario"):
    ax2.plot(
        group["Scaling Type"],
        group["Validation"],
        marker='o',
        linestyle='-'
    )
ax2.set_xlabel("Scaling Type")
ax2.set_ylabel("Accuracy")
ax2.set_title("Validation vs Scaling Type")
ax2.set_xticks(range(len(df["Scaling Type"].unique())))  # Set tick positions
ax2.set_xticklabels(['Non Scaled', 'Feature Scaled by Older Adults 1',
       'Feature Scaled by Validation', 'Raw Scaled'], rotation=45)  # Set tick labels
ax2.set_ylim(0, 1)
ax2.grid(True)

# Create a common legend above subplots
fig.legend(handles, labels, loc='lower center')
# Adjust layout and display the plot
plt.tight_layout()
plt.show()
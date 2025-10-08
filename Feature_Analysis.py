import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import kstest
plt.style.use("ggplot")
sns.set_palette("husl",8)


def features_plots_multi_exercises():
    # combine all data sources and exercises features together
    data_source = ['maya', 'naama', 'naama_pilot']
    exercises_list = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows', 'open_arms_and_forward']

    df = pd.DataFrame()
    for ds in data_source:
        for e in exercises_list:
            temp = pd.read_csv(r'C:/Users/mayak/PycharmProjects/DataAnalysis/CSV/features/old/'+ds+'_'+e+"featuresbyhand.csv" )
            temp.insert(1, "Exercise", [e]*len(temp.index), True)
            temp.insert(1, "Source", [ds]*len(temp.index), True)
            df = df.append(temp, ignore_index=True)

    feature_col = 5
    features_name = df.columns[feature_col:]
    features_amount = len(features_name)

    df['Exercise'] = df['Exercise'].replace({'raise_arms_horizontally': '1', 'bend_elbows': '2', 'raise_arms_bend_elbows':'3'})
    df['Source'] = df['Source'].replace({'maya': 'older adults', 'val1': 'young adults'})
    df['label'] = df['label'].replace({0: 'Consistent & Correct', 1: 'Non-consistent &/or Incorrect'})
    df['Combined'] = df['Source'] + ' - ' + df['label'].astype(str)

    selected_colors = sns.color_palette("husl", 10)
    selected_colors = [selected_colors[i] for i in [0, 1, 4, 7]]
    sns.set_palette(selected_colors)

    for f_name in features_name:
        feature = df[f_name]
        plt.figure()
        sns.boxplot(data=df, y=feature, x="Exercise", hue="Combined")
        plt.title(f_name)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=4)

    # Define the number of subplots per row and column
    subplots_per_row = 2
    subplots_per_col = 2

    # Calculate the number of rows needed
    num_features = len(features_name)
    num_rows = (num_features - 1) // subplots_per_row + 1

    # Create lists to store legend handles and labels
    # Iterate through features and create separate figures for each set of 2x2 subplots
    for start_idx in range(0, num_features, subplots_per_row * subplots_per_col):
        end_idx = start_idx + subplots_per_row * subplots_per_col
        subset_features = features_name[start_idx:end_idx]

        # Create a new figure with 2x2 subplots
        fig, axes = plt.subplots(subplots_per_row, subplots_per_col, figsize=(10, 8))

        # Flatten the axes array to simplify indexing
        axes = axes.flatten()

        # Iterate through the subset of features and create boxplots
        for i, f_name in enumerate(subset_features):
            feature = df[f_name]

            # Create a boxplot in the current subplot
            sns.boxplot(data=df, y=feature, x="Exercise", hue="Combined", ax=axes[i])

            axes[i].set_title(f_name)
            axes[i].set_ylabel("Value")  # Set y-axis label to "Value"
            axes[i].legend().set_visible(False)  # Remove legend from individual subplots

        # Extract the handles and labels from the first subplot
        handles, labels = axes[0].get_legend_handles_labels()

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Show the plot for this set of 2x2 subplots
        plt.show()

    # Create a single legend for the entire figure
    # fig.legend(handles=handles, labels=labels, ncol=2)

    # Display the final figure
    plt.show()


def combine_data(onlymaya = True, raw = True):
    # Combine all data sources and exercises files to one df
    if onlymaya:
        if raw:
            datasource = ['maya_raw_scaled']
            datasource = ['val1_raw_scaled']
        else:
            datasource = ['maya']
    else:
        datasource = ['maya', 'naama', 'naama_pilot', 'val1']
    exercises = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows'] #, 'open_arms_and_forward']

    df = pd.DataFrame()
    for ds in datasource:
        for e in exercises:
            temp = pd.read_csv('CSV/features/old/'+ds+"_"+e+"featuresbyhand.csv")
            temp.insert(1, "Exercise", [e]*len(temp.index), True)
            temp.insert(1, "Source", [ds]*len(temp.index), True)
            df = df.append(temp, ignore_index=True)

    df = df.dropna()  # data with only one detected cycle -> have null values in some features.
    return df


def scatter_plot():
    # Create scatter plots to see distribution by data source of the exercise's features clusters
    datasource = ['maya', 'naama', 'naama_pilot']
    ds_dict = {'maya': 'older adults 1', 'naama': 'older adults 2', 'naama_pilot': 'young adults', 'maya_raw_scaled': 'older adults 1 - raw data scaled'}
    exercises = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows', 'open_arms_and_forward']

    df = pd.read_csv('CSV/features/old/allfeaturesbyhandscaled.csv')  # for plotting scaled data (only 3 exercises
    df = pd.read_csv('CSV/features/old/allfeaturesbyhand.csv')  # for plotting not scaled data
    df = pd.read_csv('CSV/features/allfeatures_nonscaled_label.csv')  # for plotting not scaled data

    feature_col = 6
    exercises = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows']

    df = pd.read_csv('CSV/features/old/newallfeaturesbyhand.csv') # for plotting raw scaled data
    feature_col = 6

    features = df.columns[feature_col:]
    features_to_remove= ['rep_count','CL', 'cycles num']
    features = [f for f in features if f not in features_to_remove]

    sns.set_palette("husl",6)

    # Plot by data source
    for ds in datasource:
        tempdf = df[df["Source"] == ds]
        X = tempdf[features]

        # PCA
        pca = PCA()
        scores_pca = pca.fit_transform(X)
        scores_pca_df = pd.DataFrame(scores_pca)

        plt.figure()
        label = tempdf["Exercise"]
        uniq = exercises
        for i in uniq:
            plt.plot(scores_pca[label == i, 0], scores_pca[label == i, 1], ls="", marker='o', label=i)
        # plt.suptitle(cluster_colname)
        plt.title(ds_dict[ds])
        plt.legend()
        plt.show()

    sns.set_palette(sns.color_palette("husl",6)[3:])
    # Plot by exercise
    for e in exercises:
        tempdf = df[df["Exercise"] == e]
        X = tempdf[features]

        # PCA
        pca = PCA()
        scores_pca = pca.fit_transform(X)
        scores_pca_df = pd.DataFrame(scores_pca)

        plt.figure()
        label = tempdf["Source"]
        uniq = datasource
        for i in uniq:
            plt.plot(scores_pca[label == i, 0], scores_pca[label == i, 1], ls="", marker='o', label=ds_dict[i])
        # plt.suptitle(cluster_colname)
        plt.title("Data " + e)
        plt.legend()
        plt.show()


def scatter_plot_labeld():
    file_name = 'allfeaturesbyhand_label.csv'  # File containing all data without scale
    file_name = 'allfeaturesbyhandscaled_label.csv' # File containing all data features scaled
    file_name = 'newallfeaturesbyhand_label.csv' # File containing all data raw scaled
    df = pd.read_csv(f'CSV/features/old/{file_name}')
    df = df[df['Source'] == 'maya']
    features = df.columns[5:]
    features = features[1:-2]
    X = df[features]

    # PCA
    pca = PCA()
    scores_pca = pca.fit_transform(X)
    scores_pca_df = pd.DataFrame(scores_pca)

    plt.figure()
    label = df["label"]
    uniq = df["label"].unique()
    for i in uniq:
        plt.plot(scores_pca[label == i, 0], scores_pca[label == i, 1], ls="", marker='o', label=i)
    # plt.suptitle(cluster_colname)
    plt.title("older adults 1 - feature scaled by label")
    plt.legend()
    plt.show()

    # plot label + exercise
    df['labelandex'] = df['label'].astype(str) + ' ' + df['Exercise'].astype(str)

    sns.set_palette("tab10")


    label = df['labelandex']
    uniq = df['labelandex'].unique()
    plt.figure()
    colors = ['red', 'pink', 'blue', 'green', 'orange', 'yellow']

    # Set the color cycle using set_prop_cycle
    plt.gca().set_prop_cycle(color=colors)

    for i in uniq:
        plt.plot(scores_pca[label == i, 0], scores_pca[label == i, 1], ls="", marker='o', label=i)
        # plt.suptitle(cluster_colname)
        plt.title("older adults 1 - features scaled by label")
        plt.legend()
        plt.show()


def compare_dist_data_sources():
    datasource = ['maya', 'naama', 'naama_pilot']
    exercises = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows', 'open_arms_and_forward']
    df = combine_data(True)

    for e in exercises:
        print(e +'##############')
        df_ex = df[df['Exercise'] == e]
        feature_col = 5
        features = df.columns[feature_col:]

        for ds in range(0, 2):
            for ds2 in range(ds+1, 3):
                df_ex_tmp = df_ex[(df_ex["Source"] == datasource[ds]) | (df_ex["Source"] == datasource[ds2])]
                print(datasource[ds], datasource[ds2])
                for f in features:
                    df_m = df_ex_tmp[df_ex_tmp['Source'] == datasource[ds]][f]
                    df_n = df_ex_tmp[df_ex_tmp['Source'] == datasource[ds2]][f]
                    stat, p_value = kstest(df_m, df_n)
                    print(f"{f}Kolmogorov-SmirnovTest:  {stat:.4f} {p_value:.4f}")

    #
    # from joypy import joyplot
    # for f in features:
    #     joyplot(df, by='Source', column=f, colormap=sns.color_palette("crest", as_cmap=True))
    #     plt.xlabel('Income')
    #     plt.title("Ridgeline Plot, multiple groups")


def feature_hist():
    df = combine_data()
    featureColNum = 6
    features_name = df.columns[featureColNum:-2]
    for i in range(0,19):
        fig = plt.figure(constrained_layout=True, figsize=(6, 9))
        # create 3x1 subfigs
        subfigs = fig.subfigures(nrows=4, ncols=1)
        for row, subfig in enumerate(subfigs):
            f_name = features_name[i*4+row]
            feature = df[f_name]
            subfig.suptitle(f_name)
            # create 1x3 subplots per subfig
            axs = subfig.subplots(nrows=1, ncols=3)
            sns.boxplot(ax=axs[0], x="Exercise", y=f_name, data=df)
            sns.barplot(ax=axs[1], x="Exercise", y=f_name, data=df)
            sns.histplot(ax=axs[2], data=df, x=feature, kde=True, hue='Exercise', legend=False, fill=True)
            axs[1].set_xticklabels(['Ex1','Ex2','Ex3','Ex4'])
            axs[0].set_xticklabels(['Ex1','Ex2','Ex3','Ex4'])
        # plt.show()
        # plt.tight_layout()
        fig.savefig('features hist/allex_'+str(i)+'.png')


def features_stats(df_maya, features):
    result = df_maya.groupby(['Source', 'Exercise']).agg(
        **{f'{feature}_Mean': (f'{feature}', 'mean') for feature in features},
        **{f'{feature}_Std': (f'{feature}', 'std') for feature in features},
        **{f'{feature}_Min': (f'{feature}', 'min') for feature in features},
        **{f'{feature}_Max': (f'{feature}', 'max') for feature in features}
    ).reset_index()


if __name__ == '__main__':
    # scatter_plot()
    # compare_dist_data_sources()

    df = combine_data()
    exercises = ['raise_arms_horizontally', 'bend_elbows', 'raise_arms_bend_elbows', 'open_arms_and_forward']
    df_ex = df[df['Exercise'] == exercises[0]]
    df_ex_tmp = df_ex[(df_ex["Source"] == 'maya') | (df_ex["Source"] == 'val1')]

    feature_col = 6
    features = df.columns[feature_col:]
    f = features[1]
    from scipy.stats import mannwhitneyu

    for e in exercises:
        df_ex = df[df['Exercise'] == e]
        df_ex_tmp = df_ex[(df_ex["Source"] == 'maya') | (df_ex["Source"] == 'naama')]
        count = 0
        all_count = 0
        for f in features:
            feat = df_ex_tmp[f].values
            feat_m = df_ex_tmp.loc[df_ex_tmp.Source == 'maya', f].values
            feat_n = df_ex_tmp.loc[df_ex_tmp.Source == 'naama', f].values

            sample_stat = np.mean(feat_m) - np.mean(feat_n)
            stats = np.zeros(1000)
            for k in range(1000):
                labels = np.random.permutation((df_ex_tmp['Source'] == 'maya').values)
                stats[k] = np.mean(feat[labels]) - np.mean(feat[labels==False])
            p_value = np.mean(stats > sample_stat)
            # print(f"Permutation test: p-value={p_value:.4f}")

            # stat, p_value = mannwhitneyu(feat_m, feat_n)
            # print(f" Mann–Whitney U Test: statistic={stat:.4f}, p-value={p_value:.4f}")
            all_count += 1
            if p_value>0.05:
                count+=1
        print(count/all_count)
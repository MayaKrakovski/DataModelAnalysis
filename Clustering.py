import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

import seaborn as sns
plt.style.use('ggplot')
sns.set_palette("husl", 7)


def choose_file_name():
    file_name = 'allfeatures_nonscaled.csv'  # File containing all data
    file_name = 'allfeatures_scaled_label.csv'  # File containing all data
    file_name = 'allfeatures_scaledbyval1_label.csv'
    file_name = 'allfeatures_raw_scaled_label.csv'  # File containing all data

    file_name = 'ODS_YDS_allfeatures_nonscaled.csv'
    file_name = 'ODS_YDS_allfeatures_raw_scaled.csv'

    file_name = 'ODS_YDS_allfeatures_scaledbyODS.csv'  #s2
    file_name = 'ODS_YDS_all_raw_data_scaledfeaturesbyhand.csv'  #s3
    file_name = 'ODS_YDS_all_raw_datafeaturesbyhand.csv'  #s1


def read_data(file_name):
    df = pd.read_csv(f'CSV/features/{file_name}')
    df = df[df['Source'] == 'ODS'] # data of older adults set only
    return df


def PCA_explained_variance(X):
    # feature scaling: kmeans is based on distance measures, therefore need to standardize data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)
    scaled_features = pd.DataFrame(scaled_features)

    # pca
    pca = PCA()
    pca_scores = pca.fit_transform(scaled_features)

    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    # Calculate cumulative explained variance
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    # Plot cumulative explained variance
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-')
    plt.title('Cumulative Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(np.arange(1, len(cumulative_explained_variance) + 1, 4))
    print(cumulative_explained_variance)
    index_above_09 = np.argmax(cumulative_explained_variance > 0.9)
    print(f"The first {index_above_09} components explains {cumulative_explained_variance[index_above_09]*100}% of the variance - returns these only")

    pca = PCA(index_above_09)
    scores_pca = pca.fit_transform(scaled_features)
    scores_pca_df = pd.DataFrame(scores_pca)

    return scores_pca_df


def number_of_clusters(scores_pca_df):
    # number of clusters - kmeans
    plt.figure(figsize=(6, 4))
    for i in range(1, 2):
        sse = []
        for k in range(2, 15):
            kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300)
            kmeans.fit(scores_pca_df)
            sse.append(kmeans.inertia_)
        plt.plot(range(2, 15), sse)
        plt.xticks(range(2, 15))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.title("Number of Clusters - K-means")

    plt.figure(figsize=(6, 4))
    plt.title("Dendrogram")
    dend = shc.dendrogram(shc.linkage(scores_pca_df, method='ward'))


def plot_samples_by_category(df_withclusters, cat, scores_pca_df, title):
    # palette = sns.color_palette("husl")
    # c = palette[4]

    plt.figure()
    label = df_withclusters[cat]
    uniq = label.unique()
    for i in uniq:
        plt.plot(scores_pca_df.loc[label == i, 0], scores_pca_df.loc[label == i, 1], ls="", marker="o", label=i)
    plt.title(title)
    plt.legend()


def clustering(df):
    features = df.columns[6:]
    features = features[0:-2]
    print(f'number of features is {len(features)}')
    X = df[features]
    scores_pca_df = PCA_explained_variance(X)

    number_of_clusters(scores_pca_df)
    print(r"choose 5 for k-means") #s1
    print(r"choose 7 for k-means") #s2
    print(r"choose 4 for k-means")  #s3
    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300, random_state=123)
    kmeans_classifiar = kmeans.fit(scores_pca_df)
    cluster_labels = kmeans_classifiar.labels_

    cluster_df = pd.DataFrame()
    cluster_df['data_index'] = scores_pca_df.index.values
    cluster_df["k-means"] = cluster_labels

    print(r"choose 2 for hierechial")  # s1
    print(r"choose 3 for hierechial")  # s2
    print(r"choose 4 for hierechial")  # s3
    cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    cluster.fit_predict(scores_pca_df)
    cluster_df["Hierarchical"] = cluster.labels_

    df_withclusters = pd.merge(df, cluster_df, right_on='data_index', left_index=True)

    plot_samples_by_category(df_withclusters, "k-means", scores_pca_df, "Distribution by K-means")
    plot_samples_by_category(df_withclusters, "Hierarchical", scores_pca_df, "Distribution by Hierarchical Clustering")
    plot_samples_by_category(df_withclusters, "Exercise", scores_pca_df, "Distribution by Exercise Type")
    # plot_samples_by_category(df, "label", scores_pca_df, "Distribution by Labels")


def main_clustering(add_labels=True):
    file_name = 'ODS_YDS_all_raw_datafeaturesbyhand.csv'
    df = read_data(file_name)
    print(f'records_num in df {len(df)}')

    # Adding labels:
    if add_labels:
        file_name_with_label = 'allfeatures_scaled_label.csv'  # File containing all data
        df_labels = pd.read_csv(f'CSV/features/{file_name_with_label}')
        df_labels = df_labels[df_labels["Source"] == 'maya']
        select_columns = ["Exercise", "Participant", "hand", "label"]
        df_labels = df_labels[select_columns]
        df = pd.merge(df, df_labels, on=["Exercise", "Participant", "hand"])
    clustering(df)

    file_name = "ODS_YDS_allfeatures_scaledbyODS.csv"
    df = read_data(file_name)
    print(f'records_num in df {len(df)}')
    df = pd.merge(df, df_labels, on=["Exercise", "Participant", "hand"])
    clustering(df)

    file_name = "ODS_YDS_all_raw_data_scaledfeaturesbyhand.csv"
    df = read_data(file_name)
    print(f'records_num in df {len(df)}')
    clustering(df)

    plt.show()
    print("start")

    sns.set_palette("husl", 7)


if __name__ == '__main__':
    print("start")
    main_clustering()
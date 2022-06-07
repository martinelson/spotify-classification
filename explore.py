import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading in Dataset and establishing genres
genres = ['Hip Hop', 'R&B', 'Dance', 'Country', 'Blues', 'Reggae', 'Folk', 'Pop', 'Rock', 'Jazz']
df = pd.read_csv('track_features.csv')

# Dropping selected columns and establishing histogram categories
df = df.drop(['liveness', 'valence', 'type', 'uri', 'track_href', 'analysis_url', 'id'], axis=1)
hist_plot_cats = ['energy', 'danceability', 'tempo', 'loudness', 'speechiness', 'instrumentalness',
                  'acousticness', 'duration_ms']

# Plotting combined histograms
for cat in hist_plot_cats:
    plt.figure()
    sns.histplot(data=df, x=cat, hue='genre', bins=10)
    plt.savefig(f'plots/group_hist/{cat}_hist.png')
    plt.close()

# Plotting boxplot for Key category
plt.figure()
sns.boxplot(x='genre', y='key', data=df)
plt.savefig('plots/key_box.png')
plt.close()

# Ploting histogram and kde plots per genre per feature
for genre in genres:
    df_sub = df.loc[df['genre'] == genre]
    for cat in hist_plot_cats:
        plt.figure()
        sns.histplot(data=df_sub, x=cat, bins=10)
        plt.savefig(f'plots/plots_by_genre/hist/{genre}_{cat}.png')
        plt.close()

        plt.figure()
        plt.title(f'{genre} {cat}')
        df_sub[cat].plot.kde(bw_method=0.3)
        plt.savefig(f'plots/plots_by_genre/kde/{genre}_{cat}.png')
        plt.close()

# Dropping columns for correlation matrix
df_corr = df.drop(['mode', 'key', 'time_signature', 'genre', 'genre_num', 'Track Artist', 'Track Name',
                   'genre_num_combo'], axis=1)
# Plotting correlation matrix
corr = df_corr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidth=.5, cbar_kws={"shrink": 0.5})
plt.savefig('plots/correlation_matrix.png')

# Creating various dataframes for analysis.py
key_series = pd.get_dummies(df['key'], prefix='key', drop_first=True)
time_sig_series = pd.get_dummies(df['time_signature'], prefix='time_sig', drop_first=True)
genre_category = df.pop("genre_num")
genre_combo = df.pop("genre_num_combo")
df = df.drop(['key', 'time_signature', 'Track Name', 'Track Artist', 'genre'], axis=1)
df_1 = pd.concat([df, key_series, time_sig_series, genre_category], axis=1)
df_2 = pd.concat([df, genre_category], axis=1)
df_3 = pd.concat([df, genre_combo], axis=1)


def shuffle_data(df):
    cols = list(df.keys())
    data = df.to_numpy()
    np.random.seed(8)
    data = np.random.permutation(data)
    data = pd.DataFrame(data, columns=cols)
    return data


data = shuffle_data(df_1)
data_2 = shuffle_data(df_2)
data_3 = shuffle_data(df_3)
df_4 = df_2.drop(['speechiness', 'energy', 'instrumentalness', 'acousticness', 'danceability'], axis=1)
data_4 = shuffle_data(df_4)

data.to_csv('track_features_shuffled.csv', index=False)
data_2.to_csv('track_features_2_shuffled.csv', index=False)
data_3.to_csv('track_features_3_shuffled.csv', index=False)
data_4.to_csv('track_features_4_shuffled.csv', index=False)

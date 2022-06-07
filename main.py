import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

# Establishing Scope the Script can do to the Spotify Account
scope = 'playlist-read-private'

# Spotify App ID and Secret
SPOTIFY_CLIENT_ID = CLIENT_ID
SPOTIFY_CLIENT_SECRET = CLIENT_SECRET

# OAuth Settings
sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        scope="playlist-modify-private",
        redirect_uri="http://example.com",
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        show_dialog=True,
        cache_path="token.txt"
    )
)

# Getting User ID
user_id = sp.current_user()["id"]

# Getting specific playlist IDs of genres
user_playlist = sp.current_user_playlists()
genres = ['Hip Hop', 'R&B', 'Dance', 'Country', 'Blues', 'Reggae', 'Folk', 'Pop', 'Rock', 'Jazz']
genre_playlists_ids = []
for playlist in user_playlist['items']:
    if playlist['name'] in genres:
        genre_playlists_ids.append(playlist['id'])

# Creating Dataset of tracks from the 10 playlists
track_lists = dict()
track_names = []
for i in range(len(genre_playlists_ids)):
    track_id_list = []
    genre_key = genres[i]
    tracks = sp.playlist_items(genre_playlists_ids[i], limit=100)
    for track in tracks['items']:
        track_names.append([track['track']['name'], track['track']['artists'][0]['name'], track['track']['id']])
        track_id_list.append(track['track']['id'])
    track_lists[genre_key] = track_id_list

df_names = pd.DataFrame(track_names, columns=['Track Name', 'Track Artist', 'id'])

# Getting Track Features from the set of tracks
feature_list = dict()
for i in range(len(track_lists)):
    features = sp.audio_features(track_lists[genres[i]])
    feature_list[genres[i]] = features

# Creating dataframe to use in explore.py
data_df = pd.DataFrame()
for i in range(len(genres)):
    genre = genres[i]
    df = pd.DataFrame(feature_list[genre])
    df['genre'] = genre
    df['genre_num'] = i

    data_df = pd.concat([data_df, df])

result = pd.merge(data_df, df_names, on='id')

result.to_csv('track_features.csv', index=False)


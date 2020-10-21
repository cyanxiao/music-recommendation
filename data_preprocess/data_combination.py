import numpy as np
import pandas as pd
import dask.dataframe as dd


def merge_duplicate(user_artist_data: str, artist_alias: str):
    user_artist = dd.read_csv(user_artist_data, header=None, delimiter=' ', dtype={0: 'int', 1: 'int', 2: 'int'})
    user_artist.columns = ['user', 'artist', 'num']
    alias = dd.read_csv(artist_alias, header=None, delimiter='\t')
    alias = alias.fillna(-1).astype('int')  # 忽略缺少某列的情况
    alias.columns = ['alias', 'artist']
    mid = dd.merge(user_artist, alias, left_on='artist', right_on='artist', how='left')
    result = dd.merge(mid, user_artist, left_on='alias', right_on='artist', how='left')
    result = result.compute()
    print(result.iloc[0, :])


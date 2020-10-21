import numpy as np
import pandas as pd


def merge_duplicate(user_artist_data: str, artist_alias: str):
    user_artist = pd.read_csv(user_artist_data, header=None, delimiter=' ', dtype={0: 'int', 1: 'int', 2: 'int'})
    alias = pd.read_csv(artist_alias, header=None, delimiter='\t')
    alias = alias.fillna(-1).astype('int')  # 忽略缺少某列的情况

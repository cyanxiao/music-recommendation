import numpy as np
import pandas as pd
import dask.dataframe as dd


def merge(result_data: str, merged_data: str):
    user_artist = pd.read_csv(result_data, header=None, delimiter=' ', dtype={0: 'int', 1: 'int', 2: 'int'})
    user_artist.columns = ['user', 'artist', 'num']
    user_artist.groupby(['user', 'artist']).sum()
    user_artist.to_csv(merged_data)


def get_alias_to_name(artist_alias: str) -> dict:
    alias = open(artist_alias)
    line = alias.readline()
    alias_to_artist = dict()
    while line:
        try:
            line_content = line.strip().split('\t')
            alias_to_artist[int(line_content[0])] = int(line_content[1])
            line = alias.readline()
        except:
            line = alias.readline()
    alias.close()
    return alias_to_artist


def replace_alias(artist_alias: str, user_artist_data: str, result_data: str):
    _ = 0
    alias_to_artist = get_alias_to_name(artist_alias)
    user_artist = open(user_artist_data)
    lines = []
    line = user_artist.readline()
    while line:
        line_content = line.split(' ')
        if int(line_content[1]) in alias_to_artist.keys():
            _ += 1
            line_content[1] = alias_to_artist[int(line_content[1])]
            line_content = " ".join(str(x) for x in line_content)
            lines.append(line_content)
            line = user_artist.readline()
            continue
        lines.append(line)
        line = user_artist.readline()
    user_artist.close()
    result = open(result_data, 'a')
    for line in lines:
        result.write(line)
    result.close()
    print(_)
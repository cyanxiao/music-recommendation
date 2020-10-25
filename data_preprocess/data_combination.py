import numpy as np
import pandas as pd
import dask.dataframe as dd


def merge(artist_alias: str, user_artist_data: str, result_data: str, merged_data: str):
    """
    导出一个合并了 alias 和本名（全部根据 artist_alias 替换成本名）的艺术家列表，形如：
    user,artist,num
    90,24,1
    90,46,4
    90,121,1
    ......
    其中不存在 (user, artist) 的重复项，不可能同时存在  (user, artist) 相同的两行，不可能既有 (90,24,1) 又有 (90,24,2)
    :param artist_alias: artist_alias.txt 路径
    :param user_artist_data: user_artist_data.txt 路径
    :param result_data: 一个中间结果的导出路径，存在 (user, artist) 的重复项
    :param merged_data: 最终结果的导出路径，不存在 (user, artist) 的重复项
    :return:
    """
    _replace_alias(artist_alias, user_artist_data, result_data)
    user_artist = pd.read_csv(result_data, header=None, index_col=None, delimiter=' ', dtype={0: 'int', 1: 'int', 2: 'int'})
    user_artist.columns = ['user', 'artist', 'num']
    user_artist.groupby(['user', 'artist']).sum().to_csv(merged_data)


def _get_alias_to_name(artist_alias: str) -> dict:
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


def _replace_alias(artist_alias: str, user_artist_data: str, result_data: str):
    alias_to_artist = _get_alias_to_name(artist_alias)
    user_artist = open(user_artist_data)
    lines = []
    line = user_artist.readline()
    while line:
        line_content = line.split(' ')
        if int(line_content[1]) in alias_to_artist.keys():
            line_content[1] = alias_to_artist[int(line_content[1])]
            line_content = " ".join(str(x) for x in line_content)
            lines.append(line_content)
            line = user_artist.readline()
            continue
        lines.append(line)
        line = user_artist.readline()
    user_artist.close()
    result = open(result_data, 'w')
    for line in lines:
        result.write(line)
    result.close()
from scipy.sparse import *
import numpy as np
import gc


class Node(object):
    def __init__(self, char: str):
        self.char = char
        self.child = []
        self.value = 0


class TrieTree(object):
    def __init__(self):
        self.root = Node("")

    def find(self, num: str, add: bool = True) -> bool:
        ptr = self.root
        for i in range(len(num)):
            match = False
            # in child
            for nd in ptr.child:
                if nd.char.__eq__(num[i]):
                    match = True
                    ptr = nd
                    if i == len(num) - 1:
                        if ptr.value == 1:
                            return True
                        else:
                            if add:
                                ptr.value = 1
                            return False

            if not match and not add:
                return False
            if not match:
                d = Node(num[i])
                ptr.child.append(d)
                if i == len(num) - 1:
                    d.value = 1
                ptr = d
        return False

    def print_t(self):
        def dfs(nd):
            print("#")
            print(nd.char)
            print(nd.value)
            print("##")
            for d in nd.child:
                dfs(d)
        dfs(self.root)


def get_user_and_artist_maps(user_artist_data: str, output: str, save: bool = True,
                       data_sep: str = " ", result_sep: str = ",") -> None:
    """
    通过扫描数据生成用户序号和编号的双射以及艺术家序号和艺术家编号的双射，存入文件。
    :param user_artist_data:
    :param output:
    :param data_sep:
    :param result_sep:
    :return:
    """
    f = open(user_artist_data, encoding="utf-8")
    user_list = []
    artist_list = []
    art_set = set()
    first = True
    print("reading data ... ...")
    cnt = 0
    last_user = -1
    for line in f:
        if first:
            first = False
            continue
        args = line.strip().split(data_sep)
        user = int(args[0])
        artist = int(args[1])
        num = int(args[2])
        if user != last_user:
            last_user = user
            user_list.append(user)
        # if not trie.find(args[1]):
        if artist not in art_set:
            art_set.add(artist)
            artist_list.append(artist)
        cnt = cnt + 1
        if cnt % 100000 == 0:
            print("# " + str(cnt) + " was read")
    f.close()
    # del f
    # gc.collect()
    print("# " + str(cnt) + " was read")
    print()
    print("# Users Count:")
    print(str(len(user_list)))
    print("# Artists Count:")
    print(str(len(art_set)))
    print(str(len(artist_list)))
    save_list_dict(user_list, output + "user_dict.txt", sep=result_sep)
    # del art_set
    # del user_list
    # gc.collect()
    save_list_dict(artist_list, output + "artist_dict.txt", sep=result_sep)


def convert2preference(user_artist_data: str, output: str, save: bool = True,
                       data_sep: str = " ", result_sep: str = ",") -> np.ndarray:
    """
    把用户-艺术家-播放次数 表格转换成偏好矩阵。
    :param data_sep: 数据集分隔符。
    :param result_sep: 结果分隔符。
    :param save: 是否保存映射和矩阵为文件。
    :param user_artist_data:用户-艺术家-播放次数 列表的路径//名字，csv形式。
    :param output: 偏好矩阵、用户映射、艺术家映射输出目录，以//或者\\结尾，依照具体的系统而用户自定。
    :return: 返回偏好矩阵
    """
    f = open(user_artist_data, encoding="utf-8")
    user_list = []
    artist_list = []
    trie = TrieTree()
    art_dict = dict()
    user_pref = dict()
    first = True
    print("reading data ... ...")
    cnt = 0
    last_user = -1
    for line in f:
        if first:
            first = False
            continue
        args = line.strip().split(data_sep)
        user = int(args[0])
        artist = int(args[1])
        num = int(args[2])
        if user == last_user:
            user_pref[user].append((artist, num))
        else:
            last_user = user
            user_list.append(user)
            user_pref[user] = []
        # if not trie.find(args[1]):
        if artist not in art_dict.keys():
            art_dict[artist] = len(artist_list)
            artist_list.append(artist)
        cnt = cnt + 1
        if cnt % 100000 == 0:
            print("# " + str(cnt) + " was read")

    f.close()
    del f
    del trie
    gc.collect()
    print("# " + str(cnt) + " was read")
    print()
    print("# Users Count:")
    print(str(len(user_list)))
    print("# Artists Count:")
    print(str(len(art_dict.keys())))
    print()
    print("# generate preference ... ...")

    # generate preference
    cnt = 0
    preference = []
    artist_line = []
    for j in range(len(artist_list)):
        artist_line.append(-1)
    for i in range(len(user_list)):
        a = artist_line.copy()
        lst = user_pref[user_list[i]]
        for j in range(len(lst)):
            a[art_dict[lst[j][0]]] = lst[j][1]
        preference.append(a)
        cnt = cnt + 1
        if cnt % 1000 == 0:
            print("# " + str(cnt) + " users was processed")

    print("# " + str(cnt) + " users was processed")
    print()
    if save:
        preference = np.array(preference)
        save_matrix(preference, output + "preference.csv", sep=result_sep)
        save_list_dict(user_list, output + "user_dict.csv", sep=result_sep)
        save_list_dict(artist_list, output + "artist_dict.csv", sep=result_sep)

    return np.array(preference, dtype='float64')


def formalize_data(user_artist_data: str, output: str,
                   data_sep: str = ",", result_sep: str = ","):
    """
    把用户-艺术家-播放次数 编号，并且输出字典和编号后的 用户-艺术家-次数 数据。
    :param data_sep: 数据集分隔符。
    :param result_sep: 结果分隔符。
    :param user_artist_data:用户-艺术家-播放次数 列表的路径//名字。
    :param output: 偏好矩阵、用户映射、艺术家映射输出目录，以//或者\\结尾，依照具体的系统而用户自定。
    :return: 无返回值。
    """
    f = open(user_artist_data, encoding="utf-8")
    user_list = []
    artist_list = []
    art_dict = dict()
    first = True
    print("reading data ... ...")
    cnt = 0
    last_user = -1
    result_data = []
    for line in f:
        if first:
            first = False
            continue
        args = line.strip().split(data_sep)
        user = int(args[0])
        artist = int(args[1])
        if user != last_user:
            last_user = user
            user_list.append(user)
        if artist not in art_dict.keys():
            art_dict[artist] = len(artist_list)
            artist_list.append(artist)
        result_data.append(str(len(user_list) - 1) + result_sep + str(art_dict[artist]) + result_sep + args[2])
        cnt = cnt + 1
        if cnt % 100000 == 0:
            print("# " + str(cnt) + " was read")
            print("# " + str(cnt/242000) + "%")
    f.close()

    print("# " + str(cnt) + " was read")
    print()
    print("# Users Count:")
    print(str(len(user_list)))
    print("# Artists Count:")
    print(str(len(art_dict.keys())))
    print()
    print("# saving data ... ...")
    # save data
    f = open(output + "formalized_merged_data.txt", mode="w", encoding="utf-8")
    f.write("\n".join(s for s in result_data))
    f.flush()
    f.close()
    save_list_dict(user_list, output + "user_dict.txt", sep=result_sep)
    save_list_dict(artist_list, output + "artist_dict.txt", sep=result_sep)
    del f
    del user_list
    del artist_list
    del art_dict
    gc.collect()


def save_list_dict(mp: list, path: str, sep: str = ","):
    """
    Save the 1-1 dict for type int-int as a list.
    :param sep: the separator of the file, default ",".
    :param mp: the map that requires to save.
    :param path: the path + name of the file.
    :return: none.
    """
    lst = []
    blocks = []
    cnt = 0
    for i in range(len(mp)):
        lst.append(str(i) + sep + str(mp[i]))
        cnt = cnt + 1
        if cnt > 150000 or len(mp) - 1 == i:
            blocks.append("\n".join(s for s in lst))
            cnt = 0
            lst.clear()
    f = open(path, mode="w", encoding="utf-8")
    f.write("".join(s for s in blocks))
    f.flush()
    f.close()
    print("save!!!")


def save_dict(mp: dict, path: str, sep: str = ","):
    """
    Save the 1-1 dict for type int-int.
    :param sep: the separator of the file, default ",".
    :param mp: the map that requires to save.
    :param path: the path + name of the file.
    :return: none.
    """
    lst = []
    for key in mp.keys():
        lst.append(str(key) + sep + str(mp[key]))
    f = open(path, mode="w", encoding="utf-8")
    f.write("\n".join(s for s in lst))
    f.flush()
    f.close()


def read_dict(path: str, sep: str = ","):
    """
    Read 1-1 dict with type int-int.
    :param sep: the separator of the file, default ','.
    :param path: the path + name of the stored position.
    :return: the dict.
    """
    f = open(path, encoding="utf-8")
    dic = {}
    for line in f:
        args = line.strip().split(sep)
        dic[int(args[0])] = int(args[1])
    return dic


def save_matrix(matrix: np.ndarray, path: str, sep: str = ","):
    """
    Save a float64 matrix to file.
    :param matrix: the saving matrix.
    :param path: path + name.
    :param sep: the separator of the file, default ','.
    :return: null.
    """
    sb = []
    for i in range(len(matrix)):
        sb1 = []
        for j in range(len(matrix[i])):
            sb1.append(str(matrix[i][j]))
        sb.append(sep.join(s for s in sb1))
    f = open(path, mode="w", encoding="utf-8")
    f.write("\n".join(s for s in sb))
    f.flush()
    f.close()


def read_matrix(path: str, sep: str = ",") -> np.ndarray:
    """
    Read a matrix.
    :param path:
    :param sep:
    :return:
    """
    f = open(path, encoding='utf-8')
    mtxl = []
    for line in f:
        nums = line.strip().split(sep)
        ll = []
        for num in nums:
            ll.append(float(num))
        mtxl.append(ll)
    return np.array(mtxl, dtype="float64")


def get_sparse_matrix_from_file(formalized_data: str, sep: str = ",") -> coo_matrix:
    """
    把已经经过正则化的数据转换成用户偏好矩阵的稀疏矩阵表现形式 coo_matrix。
    :param sep: 文件的分隔符
    :param formalized_data: 经过转换的数据的 路径 + 文件。
    :return: 转换成的稀疏矩阵.
    """
    f = open(formalized_data, mode="r", encoding="utf-8")
    rows = []
    columns = []
    data = []
    cnt = 0
    last = -1
    artist_cnt = -1
    for line in f:
        args = line.strip().split(sep)
        user = int(args[0])
        artist = int(args[1])
        if last != user:
            last = user
            cnt = cnt + 1
        if artist > artist_cnt:
            artist_cnt = artist
        rows.append(user)
        columns.append(artist)
        data.append(int(args[2]))
    f.close()
    del f
    gc.collect()
    return coo_matrix((rows, columns, data), (cnt, artist_cnt + 1), dtype=float)

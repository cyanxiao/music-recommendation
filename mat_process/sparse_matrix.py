import scipy.sparse as ss
import numpy as np
import pandas as pd


def _read_formalized_data(formalized_data_path: str) -> ss.csr_matrix:
    """
    将 formalized_data 转换为稀疏矩阵
    :param formalized_data_path: formalized_data 路径
    :return: 稀疏矩阵
    """
    formalized_data = pd.read_csv(formalized_data_path, header=None, index_col=None, delimiter=',',
                                  dtype={0: 'int', 1: 'int', 2: 'int'})
    formalized_data.columns = ['X', 'Y', 'I']
    sparse_formalized_data = ss.coo_matrix((formalized_data.I, (formalized_data.X, formalized_data.Y)))
    return sparse_formalized_data.tocsr()


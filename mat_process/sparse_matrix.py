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


def read_formalized_data_to_dense_mat(formalized_data_path: str) -> np.ndarray:
    """
        将 formalized_data 转换为稀疏矩阵
        :param formalized_data_path: formalized_data 路径
        :return: 稀疏矩阵
        """
    formalized_data = pd.read_csv(formalized_data_path, header=None, index_col=None, delimiter=',',
                                  dtype={0: 'int', 1: 'int', 2: 'int'})
    formalized_data.columns = ['X', 'Y', 'I']
    sparse_formalized_data = ss.coo_matrix((formalized_data.I, (formalized_data.X, formalized_data.Y)))
    sparse_formalized_data = sparse_formalized_data.toarray().astype("float")
    sparse_formalized_data[sparse_formalized_data == 0] = np.nan
    return sparse_formalized_data


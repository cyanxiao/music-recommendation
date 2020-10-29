import numpy as np
import scipy.sparse as ss


def decompose(user_preference: np.ndarray, latent_factor_num: int = 1) -> (np.ndarray, np.ndarray):
    """
    分解用户偏好矩阵至两个初始化矩阵 U 和 V，U 的列数和 V 的行数为 latent_factor_num
    :param user_preference: 分解用户偏好矩阵
    :param latent_factor_num: U 的列数和 V 的行数
    :return: U, V
    """
    rows_num, column_num = user_preference.shape
    not_nan_elements_num = np.count_nonzero(~np.isnan(user_preference))
    avg = np.nansum(user_preference) / not_nan_elements_num
    init_element = np.sqrt(avg / latent_factor_num)
    u_init, v_init = np.full((rows_num, latent_factor_num), init_element), \
                     np.full((latent_factor_num, column_num), init_element)
    return u_init, v_init


def sparse_mat_decompose(user_preference: ss.csr_matrix, latent_factor_num: int = 1) -> (ss.csr_matrix, ss.csr_matrix):
    """
    分解稀疏的用户偏好矩阵至两个初始化矩阵 U 和 V，U 的列数和 V 的行数为 latent_factor_num
    :param user_preference: 分解用户偏好矩阵
    :param latent_factor_num: U 的列数和 V 的行数
    :return: U, V
    """
    rows_num, column_num = user_preference.shape
    not_nan_elements_num = user_preference.count_nonzero()
    avg = user_preference.sum() / not_nan_elements_num
    init_element = np.sqrt(avg / latent_factor_num)
    u_init, v_init = np.full((rows_num, latent_factor_num), init_element), \
                     np.full((latent_factor_num, column_num), init_element)
    return u_init, v_init


def get_rmse(u_mat: np.ndarray, v_mat: np.ndarray, user_preference: np.ndarray) -> np.float64:
    """
    计算 RMSE
    :param u_mat: U
    :param v_mat: V
    :param user_preference: 用户偏好矩阵
    :return: RMSE
    """
    expected_mat = np.dot(u_mat, v_mat)  # U * V
    residue_mat = np.full([user_preference.shape[0], user_preference.shape[1]], 0, "float")
    for i in range(user_preference.shape[0]):
        for j in range(user_preference.shape[1]):
            if np.isnan(user_preference[i, j]):
                residue_mat[i, j] = 0  # 忽略 NaN
            else:
                residue_mat[i, j] = user_preference[i, j] - expected_mat[i, j]
    return np.sqrt(np.sum(residue_mat ** 2) / np.size(residue_mat))

import numpy as np


def decompose(user_preference: np.ndarray, latent_factor_num: int = 1) -> (np.ndarray, np.ndarray):
    """
    分解用户偏好矩阵至两个初始化矩阵 U 和 V，U 的列数和 V 的行数为 latent_factor_num
    :param user_preference: 分解用户偏好矩阵
    :param latent_factor_num: U 的列数和 V 的行数
    :return: U, V
    """
    rows_num, column_num = user_preference.shape
    avg = np.sum(user_preference) / np.size(user_preference)
    init_element = np.sqrt(avg / latent_factor_num)
    return np.full((rows_num, latent_factor_num), init_element), \
           np.full((latent_factor_num, rows_num), init_element)


def get_rmse(u_mat: np.ndarray, v_mat: np.ndarray, user_preference: np.ndarray) -> np.float64:
    """
    计算 RMSE
    :param u_mat: U
    :param v_mat: V
    :param user_preference: 用户偏好矩阵
    :return: RMSE
    """
    residue_mat = user_preference - np.dot(u_mat, v_mat)
    return np.sqrt(np.sum(residue_mat ** 2) / np.size(residue_mat))

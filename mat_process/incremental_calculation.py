import numpy as np
from mat_process import uv_decomposition

"""
增量计算方法 1
Deprecated
"""


def _get_x_in_u(u_single_line_mat: np.ndarray, v_mat: np.ndarray,
                user_preference_single_line: np.ndarray) -> np.ndarray:
    """
    对 U 矩阵的某一行进行计算，得到一行的增量计算结果（解方程）
    :param u_single_line_mat: U 的某一行
    :param v_mat: V
    :param user_preference_single_line: 用户偏好矩阵的某一行
    :return: U 矩阵该行增量计算后的结果
    """
    assert u_single_line_mat.ndim == 1, 'u_single_line_mat row number overflows.'
    assert user_preference_single_line.ndim == 1, 'user_preference_single_line row number overflows.'
    new_u_single_line_mat = u_single_line_mat  # 待计算的 U 的新行
    for i in range(u_single_line_mat.shape[0]):
        new_u_single_line_mat[i] = 0
        tmp_user_preference_single_line = np.dot(new_u_single_line_mat, v_mat)
        other_sum = np.nansum(user_preference_single_line - tmp_user_preference_single_line)
        coefficient_x = _get_coefficient_x(v_mat[i, :], user_preference_single_line)
        x = other_sum / coefficient_x
        new_u_single_line_mat[i] = x
    return new_u_single_line_mat


def _get_coefficient_x(v_single_line_mat: np.ndarray, user_preference_single_line: np.ndarray) -> float:
    """
    获取在 user_preference_single_line 中有 NaN 时未定数的系数（仅在 _get_x_in_u() 中使用）
    """
    nan_positions = np.argwhere(np.isnan(user_preference_single_line))
    coefficient = 0.0
    for i in range(v_single_line_mat.shape[0]):
        if i in nan_positions:
            continue
        coefficient += v_single_line_mat[i]
    return coefficient


"""
增量计算方法 2
"""


def get_x(u: np.ndarray, v: np.ndarray, m: np.ndarray, r: int, s: int) -> int:
    """
    获取 U 矩阵在 (r, s) 位置的元素的值，使 M 与 UV 间的 RMSE 最小
    :param u: U
    :param v: V
    :param m: M
    :param r: r
    :param s: s
    :return: U(r, s) 的值
    """
    assert u.shape[1] == v.shape[0], 'u & v can not be multiplied.'
    assert u.shape[0] == m.shape[0] and v.shape[1] == m.shape[1], 'the shape of m/u/v needs to be revised.'
    numerator = 0.0
    denominator = 0.0
    for j in range(v.shape[1]):
        sum_of_u_v = np.dot(u[r, :], v[:, j]) - u[r, s] * v[s, j]
        single_numerator = v[s, j] * (m[r, j] - sum_of_u_v)
        if np.isnan(m[r, j]):
            continue
        numerator += single_numerator
        denominator += v[s, j] ** 2
    return numerator / denominator


def get_y(u: np.ndarray, v: np.ndarray, m: np.ndarray, r: int, s: int) -> int:
    """
    获取 V 矩阵在 (r, s) 位置的元素的值，使 M 与 UV 间的 RMSE 最小
    :param u: U
    :param v: V
    :param m: M
    :param r: r
    :param s: s
    :return: U(r, s) 的值
    """
    assert u.shape[1] == v.shape[0], 'u & v can not be multiplied.'
    assert u.shape[0] == m.shape[0] and v.shape[1] == m.shape[1], 'the shape of m/u/v needs to be revised.'
    numerator = 0.0
    denominator = 0.0
    for i in range(u.shape[0]):
        sum_of_u_v = np.dot(u[i, :], v[:, s]) - u[i, r] * v[r, s]
        single_numerator = u[i, r] * (m[i, s] - sum_of_u_v)
        if np.isnan(m[i, s]):
            continue
        numerator += single_numerator
        denominator += u[i, r] ** 2
    return numerator / denominator


def process(user_preference_mat: np.ndarray, latent_factor_num: int) -> (np.ndarray, float):
    """
    获取增量计算分解后的用户偏好矩阵和 RMSE
    :param user_preference_mat: 初始用户偏好矩阵
    :param latent_factor_num: 潜在因子数量
    :return: 补全后的用户偏好矩阵，RMSE
    """
    u, v = uv_decomposition.decompose(user_preference_mat, latent_factor_num)
    for r in range(u.shape[0]):
        for s in range(u.shape[1]):
            u[r, s] = get_x(u, v, user_preference_mat, r, s)
    for r in range(v.shape[0]):
        for s in range(v.shape[1]):
            v[r, s] = get_y(u, v, user_preference_mat, r, s)
    filled_user_preference_mat = np.dot(u, v)
    for r in range(user_preference_mat.shape[0]):
        for s in range(user_preference_mat.shape[1]):
            if np.isnan(user_preference_mat[r, s]):
                user_preference_mat[r, s] = filled_user_preference_mat[r, s]
    return user_preference_mat, uv_decomposition.get_rmse(u, v, user_preference_mat)

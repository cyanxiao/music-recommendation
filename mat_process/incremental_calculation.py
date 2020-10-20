import numpy as np


def get_x_in_u(u_single_line_mat: np.ndarray, v_mat: np.ndarray, user_preference_single_line: np.ndarray) -> np.ndarray:
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
        # coefficient_x = np.sum(v_mat[i, :])
        # TODO: 需要进一步检查生成矩阵
        coefficient_x = _get_coefficient_x(v_mat[i, :], user_preference_single_line)
        # print('coefficient_x', coefficient_x)
        # print('_get_coefficient_x', _get_coefficient_x(v_mat[i, :], user_preference_single_line))
        x = other_sum / coefficient_x
        new_u_single_line_mat[i] = x
    return new_u_single_line_mat


def _get_coefficient_x(v_single_line_mat: np.ndarray, user_preference_single_line: np.ndarray) -> float:
    """
    获取在 user_preference_single_line 中有 NaN 时未定数的系数（仅在 get_x_in_u() 中使用）
    """
    nan_positions = np.argwhere(np.isnan(user_preference_single_line))
    coefficient = 0.0
    for i in range(v_single_line_mat.shape[0]):
        if i in nan_positions:
            continue
        coefficient += v_single_line_mat[i]
    return coefficient

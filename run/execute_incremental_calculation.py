from mat_process import uv_decomposition
from mat_process import incremental_calculation
from data_preprocess import data_combination
from data_preprocess import data_convert
from mat_process import sparse_matrix
from path import data  # 数据文件存储路径

"""
非稀疏矩阵和用户偏好矩阵偏小时
"""

user_preference_mat = sparse_matrix.read_formalized_data_to_dense_mat(data.little_samples)  # 读入数据为稀疏矩阵格式
print(incremental_calculation.process(user_preference_mat, 2))  # 进行增量计算并打印




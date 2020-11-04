import numpy as np
import mat_process.uv_decomposition as uvd
from scipy.sparse import *


def rmse(user_preference: np.ndarray, estimate: np.ndarray) -> np.float64:
    return np.sqrt(np.sum((user_preference - estimate) ** 2) / np.size(user_preference))


def rmse_delta(delta: np.ndarray) -> np.float64:
    return np.sqrt(np.sum(delta ** 2) / np.size(delta))


# gradient descend
def gradient_desc_uv(user_preference: np.ndarray, latent_factor: int = 1,
                     max_iter: int = 100, step: float = 0.01, lamda: float = 0,
                     least: float = 0.001, print_msg: bool = False) -> np.ndarray:
    """
    UV decomposition with gradient descend methods.
    :param print_msg: if should print rmse in each iteration
    :param least: the least change of each iteration.
    :param lamda: the format term.
    :param step: the learning rate.
    :param max_iter: maximum iterate times.
    :param user_preference: the user preference matrix.
    :param latent_factor: the number of the latent factors.
    :return: return the estimated results of user preference matrix with empty blank filled.
    """
    u, v = uvd.decompose(user_preference, latent_factor)
    last = uvd.get_rmse(u, v, user_preference)
    optimal = last
    opt_u = u.copy()
    opt_v = v.copy()
    if print_msg:
        print("rmse 0: " + str(last))
    has_item = [[], []]
    for i in range(len(user_preference)):
        for j in range(len(user_preference[i])):
            if np.isnan(user_preference[i][j]):
                continue
            has_item[0].append(i)
            has_item[1].append(j)
    for i in range(max_iter):
        # k is the delta of user preference and UV
        k = np.zeros(shape=user_preference.shape, dtype=user_preference.dtype)
        for j in range(len(has_item[0])):
            k[has_item[0][j]][has_item[1][j]] = \
                user_preference[has_item[0][j]][has_item[1][j]] - u[has_item[0][j], :].dot(v[:, has_item[1][j]])
        u = u + step * (k.dot(v.T) - lamda * u)
        v = v + step * (u.T.dot(k) - lamda * v)
        now = uvd.get_rmse(u, v, user_preference)
        if print_msg:
            print("rmse" + str(i + 1) + ": " + str(now))
        if now < optimal:
            optimal = now
            opt_u = u.copy()
            opt_v = v.copy()
        if abs(now - last) < least:
            break
        last = now
    print(opt_u)
    print(opt_v)
    return opt_u.dot(opt_v)


def newton_uv(user_preference: np.ndarray, latent_factor: int = 1) -> np.ndarray:
    """
    UV decomposition with gradient descend methods.
    :param user_preference: the user preference matrix.
    :param latent_factor: the number of the latent factors.
    :return: return the estimated results of user preference matrix with empty blank filled.
    """
    return None


def get_delta(preference: csr_matrix, u: csr_matrix, v: csr_matrix) -> csr_matrix:
    nonzero = preference.nonzero()
    data = []
    for i in range(len(nonzero[0])):
        data.append(preference[nonzero[0][i], nonzero[1][i]] - (u[nonzero[0][i], :].dot(v[:, nonzero[1][i]])[0, 0]))
    coo = coo_matrix((data, (nonzero[0], nonzero[1])), shape=preference.shape, dtype=preference.dtype)
    return coo.tocsr()


def get_rmse_sparse(delta: csr_matrix) -> np.float64:
    nonzero = delta.nonzero()
    gross = 0
    div = np.size(delta)
    for i in range(len(nonzero[0])):
        sqr = delta[nonzero[0][i], nonzero[1][i]] ** 2
        gross = gross + sqr
    return np.sqrt(gross / div)


def gradient_desc_uv_sparse(user_preference: csr_matrix, latent_factor: int = 1,
                            max_iter: int = 100, step: float = 0.01, lamda: float = 0,
                            least: float = 0.001, print_msg: bool = False) -> (csr_matrix, csr_matrix):
    """

    :param user_preference:
    :param latent_factor:
    :param max_iter:
    :param step:
    :param lamda:
    :param least:
    :param print_msg:
    :return:
    """
    if print_msg:
        print("# begin")
    u, v = uvd.sparse_mat_decompose(user_preference, latent_factor)
    last = float('inf')
    optimal = last
    u_opt = u.copy()
    v_opt = v.copy()
    for i in range(max_iter):
        k = get_delta(user_preference, u, v)  # k is the delta of user preference and UV
        err = get_rmse_sparse(k)
        if print_msg:
            print(str(i) + ": " + str(err))
        if np.abs(err - last) < least or last - err < 0:
            break
        last = err
        if err < optimal:
            optimal = err
            u_opt = u.copy()
            v_opt = v.copy()
        u = u + step * (k.dot(v.T) - lamda * u)
        v = v + step * (u.T.dot(k) - lamda * v)
    return u_opt, v_opt

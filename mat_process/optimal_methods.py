import numpy as np
import mat_process.uv_decomposition as uvd
import scipy as scp


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
    o = user_preference.copy()
    for i in range(max_iter):
        e = u.dot(v)
        for j in range(len(o)):
            for b in range(len(o[j])):
                if np.isnan(o[j][b]):
                    o[j][b] = e[j][b]
        k = o - e  # k is the delta of user preference and UV
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
    return np.array([])


def gradient_desc_uv_sm(user_preference: np.ndarray, latent_factor: int = 1,
                     max_iter: int = 100, step: float = 0.01, lamda: float = 0,
                     least: float = 0.001, print_msg: bool = False) -> (np.ndarray, np.ndarray):

    """
    Decompose the sparse matrix.
    :param user_preference:
    :param latent_factor:
    :param max_iter:
    :param step:
    :param lamda:
    :param least:
    :param print_msg:
    :return: return U and V as np.ndarray.
    """
    return None
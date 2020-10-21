import numpy as np
import mat_process.uv_decomposition as uvd


# gradient descend
def gradient_desc_uv(user_preference: np.ndarray, latent_factor: int = 1,
                     max_iter: int = 100, step: float = 0.1, lamda: float = 0,
                     least: float = 0.1) -> np.ndarray:
    """
    UV decomposition with gradient descend methods.
    :param least: the least change of each iterate.
    :param lamda: the format term.
    :param step: the learning rate.
    :param max_iter: maximum iterate times.
    :param user_preference: the user preference matrix.
    :param latent_factor: the number of the latent factors.
    :return: return the estimated results of user preference matrix with empty blank filled.
    """
    u, v = uvd.decompose(user_preference, latent_factor)
    last = uvd.get_rmse(u, v, user_preference)
    o = user_preference.copy()
    for i in range(max_iter):
        e = u.dot(v)
        for j in range(len(o)):
            for b in range(len(o[j])):
                if np.isnan(o[j][b]):
                    o[j][b] = e[j][b]
        k = o - e  # k is the delta of user preference and UV
        u = u + step * (k.dot(v.T) - lamda * u)
        v = v + step * (k.dot(u.T) - lamda * v)
        now = uvd.get_rmse(u, v, user_preference)
        if abs(now - last) < least:
            break
        last = now
    return u.dot(v)


def newton_uv(user_preference: np.ndarray, latent_factor: int = 1) -> np.ndarray:
    """
    UV decomposition with gradient descend methods.
    :param user_preference: the user preference matrix.
    :param latent_factor: the number of the latent factors.
    :return: return the estimated results of user preference matrix with empty blank filled.
    """
    return None
import numpy as np
import mat_process.uv_decomposition as uvd


# gradient descend
def gradient_desc_uv(user_preference: np.ndarray, latent_factor: int = 1,
                     max_iter: int = 100, step: float = 0.1) -> np.ndarray:
    """
    UV decomposition with gradient descend methods.
    :param step:
    :param max_iter: maximum iterate times
    :param user_preference: the user preference matrix.
    :param latent_factor: the number of the latent factors.
    :return: return the estimated results of user preference matrix with empty blank filled.
    """
    u, v = uvd.decompose(user_preference, latent_factor)
    for i in range(max_iter):
        continue

    return u.dot(v)


def newton_uv(user_preference: np.ndarray, latent_factor: int = 1) -> np.ndarray:
    """
    UV decomposition with gradient descend methods.
    :param user_preference: the user preference matrix.
    :param latent_factor: the number of the latent factors.
    :return: return the estimated results of user preference matrix with empty blank filled.
    """
    return None
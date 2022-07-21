import numpy as np
import numba


@numba.jit(nopython=True)
def most_probable_sequence(f: np.array):
    """ Find the most probable sequence using dynamic programming
        Compute \max_{y_1,...,y_{n+1}} \sum_{i=1}^{n} f_i(y_i, y_{i+1})

    Parameters
    ----------
    f : np.ndarray
        Matrix of size (n, Y, Y).

    Returns
    -------
    sequence : np.ndarray
        Sequence of size (n + 1)
    """
    f = np.copy(f).astype(np.float64)
    
    n = f.shape[0]
    Y = f.shape[1]

    I = np.zeros((Y, n), dtype=np.int64)
    F = np.zeros((Y, n), dtype=np.float64)

    for i in range(n):
        for k in range(Y):
            distances = f[i, :, k]
            if i > 0:
                distances += F[:, i - 1]
            maximizer = distances.argmax()

            I[k, i] = maximizer
            F[k, i] = distances[maximizer]

    idx = F[:, -1].argmax(0)
    length = F[idx, -1]

    sequence = [idx]
    # for i in reversed(range(n)):
    for i in range(n - 1, -1, -1):
        idx = I[idx, i]
        sequence.insert(0, idx)
    sequence = np.array(sequence)
    
    return length, sequence


if __name__ == '__main__':
    from utils import generate_random
    
    f, y = generate_random()
    length, sequence = most_probable_sequence(f)
    
    print(sequence)
    print(length)
    print(y)
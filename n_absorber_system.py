import argparse
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def MLKF_ndof(m1, l1, k1, f1, n, m2_total, l2, k2_func):
    # n = number of absorbers

    # Create a n by n matrix
    m_n = m2_total / n
    m_trace = [m_n for i in range(n+1)]
    m_trace[0] = m1
    M = np.diag(m_trace)
    
    l_trace = [l2 for i in range(n+1)]
    l_trace[0] = l1+l2*(n+2)
    L = np.diag(l_trace)
    L[:,0] -= l2
    L[0,:] -= l2

    k_trace = np.array([k2_func(i) for i in range(n)])
    k_trace = np.append(0, k_trace)
    K = np.diag(k_trace)
    K[:,0] -= k_trace
    K[0,:] -= k_trace
    K[0, 0] = k1 + np.sum(k_trace)
    
    F = np.zeros(n+1)
    F[0] = f1

    return M, L, K, F

def freq_response(w_list, M, L, K, F):

    """Return complex frequency response of system"""

    return np.array(
        [np.linalg.solve(-w*w * M + 1j * w * L + K, F) for w in w_list]
    )

def time_response(t_list, M, L, K, F):

    """Return time response of system"""

    mm = M.diagonal()

    def slope(t, y):
        xv = y.reshape((2, -1))
        a = (F - L@xv[1] - K@xv[0]) / mm
        s = np.concatenate((xv[1], a))
        return s

    solution = scipy.integrate.solve_ivp(
        fun=slope,
        t_span=(t_list[0], t_list[-1]),
        y0=np.zeros(len(mm) * 2),
        method='Radau',
        t_eval=t_list
    )

    return solution.y[0:len(mm), :].T

def last_nonzero(arr, axis, invalid_val=-1):

    """Return index of last non-zero element of an array"""

    mask = (arr != 0)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def plot(hz, sec, M, L, K, F):

    # Generate response data

    f_response = freq_response(hz * 2*np.pi, M, L, K, F)
    f_amplitude = np.abs(f_response)
    t_response = time_response(sec, M, L, K, F)

    # Determine suitable legends

    f_legends = (
        'm{} peak {:.4g} metre at {:.4g} Hz'.format(
            i+1,
            f_amplitude[m][i],
            hz[m]
        )
        for i, m in enumerate(np.argmax(f_amplitude, axis=0))
    )

    equilib = np.abs(freq_response([0], M, L, K, F))[0]         # Zero Hz
    toobig = abs(100 * (t_response - equilib) / equilib) >= 2
    lastbig = last_nonzero(toobig, axis=0, invalid_val=len(sec)-1)

    t_legends = (
        'm{} settled to 2% beyond {:.4g} sec'.format(
            i+1,
            sec[lastbig[i]]
        )
        for i, _ in enumerate(t_response.T)
    )

    # Create plot

    fig, ax = plt.subplots(1, 2)
    
    ax[0].set_title('Amplitude of frequency domain response to sinusoidal force')
    ax[0].set_xlabel('Frequency/hertz')
    ax[0].set_ylabel('Amplitude/metre')
    ax[0].legend(ax[0].plot(hz, f_amplitude), f_legends)

    ax[-1].set_title('Time domain response to step force')
    ax[-1].set_xlabel('Time/second')
    ax[-1].set_ylabel('Displacement/metre')
    ax[-1].legend(ax[-1].plot(sec, t_response), t_legends)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    m1, l1, k1, f1 = 1, 1, 1, 1
    n = 3
    m2_total = 0.15
    l2 = 0.1
    def k2_func(i):
        return 2*i + 1

    M, L, K, F = MLKF_ndof(m1, l1, k1, f1, n, m2_total, l2, k2_func)

    hz = np.linspace(0, 5, 10001)
    sec = np.linspace(0, 30, 10001)
    plot(hz, sec, M, L, K, F)
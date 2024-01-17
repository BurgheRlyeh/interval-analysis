import numpy as np
from numpy.linalg import inv, norm
from intvalpy import Interval



def sti(x):
    result = np.zeros(2 * len(x))
    for i in range(len(x)):
        result[i] = -x[i].inf
        result[i + len(x)] = x[i].sup
    return result

def sti_inv(x):
    return [Interval(-x[i].mid, x[i + len(x) // 2].mid) for i in range(len(x) // 2)]


def partx_neg(x):
    return -1 if x < 0 else -0.5 if x == 0 else 0

def partmax_1(C, i, j, x):
    n = len(x) // 2
    prod_1 = max(0, C[i, j].sup) * max(0, x[j])
    prod_2 = max(0, -C[i, j].inf) * max(0, x[j + n])
    return (max(0, C[i, j].sup), 0) if prod_1 > prod_2 else (0.5 * max(0, C[i, j].sup), 0.5 * max(0, -C[i, j].inf)) if prod_1 == prod_2 else (0, max(0, -C[i, j].inf))

def partmax_2(C, i, j, x):
    n = len(x) // 2
    prod_1 = max(0, C[i, j].sup) * max(0, x[j + n])
    prod_2 = max(0, -C[i, j].inf) * max(0, x[j])
    return (0, max(0, C[i, j].sup)) if prod_1 > prod_2 else (0.5 * max(0, -C[i, j].inf), 0.5 * max(0, C[i, j].sup)) if prod_1 == prod_2 else (max(0, -C[i, j].inf), 0)

def partF(C, i, x):
    n = len(x) // 2
    res = np.zeros(2 * n)
    if 1 <= i <= n:
        for j in range(n):
            temp = partmax_1(C, i, j, x)
            res_1 = max(0, C[i, j].inf) * partx_neg(x[j]) + max(0, -C[i, j].sup) * partx_neg(x[j + n]) - temp[0]
            res_2 = max(0, C[i, j].inf) * partx_neg(x[j]) + max(0, -C[i, j].sup) * partx_neg(x[j + n]) - temp[1]
            res[j] -= res_1
            res[j + n] -= res_2
    else:
        i -= n
        for j in range(n):
            temp = partmax_2(C, i, j, x)
            res_1 = temp[0] - max(0, C[i, j].inf) * partx_neg(x[j + n]) - max(0, -C[i, j].sup) * partx_neg(x[j])
            res_2 = temp[1] - max(0, C[i, j].inf) * partx_neg(x[j + n]) - max(0, -C[i, j].sup) * partx_neg(x[j])
            res[j] += res_1
            res[j + n] += res_2
    return res

def D(C, x):
    n = len(x)
    D_matrix = np.zeros((n, n))
    for i in range(n):
        D_matrix[i, :] = partF(C, i + 1, x)
    return D_matrix

def init_point(C, d):
    midC = np.array([[interval.mid() for interval in row] for row in C])
    C̃ = np.array([[max(0, midC[i, j]) if i == j else max(0, -midC[i, j]) for j in range(len(midC[i]))] for i in range(len(midC))])
    return np.linalg.solve(C̃, sti(d))

def sub_diff(C, d, x0, eps):
    def g(x):
        x_inv = sti_inv(x)
        C_x = np.dot(C, sti(x_inv))
        return sti(C_x) - sti(d)

    x = x0
    g_val = g(x)
    count = 0

    while norm(g_val) >= eps:
        print("x", x)
        try:
            x -= inv(D(C, x)) @ g_val
        except:
            print("Subgradient D is singular")
            break
        g_val = g(x)
        count += 1

    return (sti_inv(x), count)


def main():
    # Пример использования
    C = np.array([[Interval(2, 4), Interval(-2, 1)],
                   [Interval(-1, 2), Interval(2, 4)]])
    d = np.array([Interval(2, 2), Interval(2, 2)])
    x0 = np.array([Interval(0, 1), Interval(0, 1)])
    eps = 1e-6
    result = sub_diff(C, d, x0, eps)
    print(result)



if __name__ == '__main__':
    main()
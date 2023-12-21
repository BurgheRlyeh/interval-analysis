import intvalpy as ip
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def create_L(X):
    midL = np.zeros((2, 2))
    radL = np.zeros((2, 2))
    radL[0][0] = (X.b[0] * 2 - X.a[0] * 2) / 2
    midL[0][0] = X.a[0] * 2 + radL[0][0]

    radL[0][1] = (X.b[1] * 2 - X.a[1] * 2) / 2
    midL[0][1] = X.a[1] * 2 + radL[0][1]

    radL[1][0] = (X.b[0] * 2 - X.a[0] * 2) / 2
    midL[1][0] = X.a[0] * 2 + radL[0][0]

    midL[1][1] = -1
    radL[1][1] = 0
    return ip.Interval(midL, radL, midRadQ=True)


def inv_midL(L):
    midL = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            midL[i][j] = L.mid[i][j]
    return np.linalg.inv(midL)


def eig_midL(L):
    midL = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            midL[i][j] = L.mid[i][j]
    return np.linalg.eig(midL)


def F_x(x):
    F = np.zeros(2)
    F[0] = x[0] ** 2 + x[1] ** 2 - 1
    F[1] = x[0] ** 2 - x[1]
    return F


def kravchik(x_av, F, Lambda, I, X, L):
    return x_av - Lambda @ F + (I - Lambda @ L) @ (X - x_av)


def intersection(X, kr):
    midL = np.zeros(2)
    radL = np.zeros(2)
    a1 = kr.a[0] if (kr.a[0] >= X.a[0]) else X.a[0]
    b1 = kr.b[0] if (kr.b[0] <= X.b[0]) else X.b[0]
    a2 = kr.a[1] if (kr.a[1] >= X.a[1]) else X.a[1]
    b2 = kr.b[1] if (kr.b[1] <= X.b[1]) else X.b[1]

    radL[0] = (b1 - a1) / 2
    midL[0] = a1 + radL[0]
    radL[1] = (b2 - a2) / 2
    midL[1] = a2 + radL[1]

    return ip.Interval(midL, radL, midRadQ=True)


def main():
    midX = [0.5, 0.5]
    radX = [0.5, 0.5]
    I = [[1, 0], [0, 1]]
    X = ip.Interval(midX, radX, midRadQ=True)

    X_k = []
    X_k.append(X)
    i = 0
    print("Количевство итераций:", i, "\n", X_k[i][1], X_k[i][0], )
    for i in range(1, 10):
        L = create_L(X)
        Lambda = inv_midL(L)
        q = eig_midL(I - Lambda @ L)
        x_av = X.mid
        F = F_x(x_av)
        kr = kravchik(x_av, F, Lambda, I, X, L)
        X_k.append(intersection(X, kr))
        print("Количевство итераций:", i, "\n", X_k[i][1], X_k[i][0], )
        X = X_k[i]

    fig, ax = plt.subplots(figsize=(4, 4))

    for i in range(10):
        one = abs(X_k[i][0].b - X_k[i][0].a)
        two = abs(X_k[i][1].b - X_k[i][1].a)
        iveRect = plt.Rectangle((X_k[i][1].a, X_k[i][0].a), two, one, edgecolor='black', facecolor='none',
                                label='Брус ive', linewidth=1.5)
        plt.gca().add_patch(iveRect)
    x = np.arange(-2, 3.01, 0.01)
    y = np.sqrt(1 - pow(x, 2))
    plt.plot(x, y, '--', linewidth=0.7)
    y = np.sqrt(x)
    plt.plot(x, y, linewidth=0.7)
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

    x = -0.5 + np.sqrt(1.25)
    y = np.sqrt(x)
    print(X_k[9][1].rad, X_k[9][0].rad)
    print("{}".format(abs(X_k[9][0].a - y)), "{}".format(abs(X_k[9][1].a - x)))
    print("{}".format(abs(X_k[9][0].b - y)), "{}".format(abs(X_k[9][1].b - x)))
    # for i in range(10):
    # print("{}".format(abs(X_k[i][0].a - y)), "{}".format(abs(X_k[i][1].a - x)))


if __name__ == "__main__":
    main()

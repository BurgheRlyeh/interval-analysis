from typing import List
import math as m
from interval import *


def det(A: IntMatrix):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def min_delta_search(A: Matrix, delta: Number, radCoeffs: Matrix) -> bool:
    def f(A: Matrix, delta: Number, radCoeffs: Matrix):
        radA = [[delta * radCoeffs[0][0], delta * radCoeffs[0][1]],
                [delta * radCoeffs[1][0], delta * radCoeffs[1][1]]]
        A_int = midrad(A, radA)
        det_A = det(A_int)
        if 0.0 in det_A:
            return True
        else:
            return False

    min_delta = delta
    a = 0
    b = delta
    while not m.isclose(a, b, rel_tol=1e-14):
        min_delta = (a + b) / 2
        if f(A, min_delta, radCoeffs):
            b = min_delta
        else:
            a = min_delta
    return (a + b) / 2


def main():
    print("Enter A_11, A_12, A_21, A_22: ", end='')
    a, b, c, d = [float(n) for n in input().split()]
    if (a < 0 or b < 0 or c < 0 or d < 0):
        raise ValueError("Values must be non-negative.")
    delta = min(a, b, c, d)
    print(delta)
    A_orig = [[a, b], [c, d]]

    ############# rad томографии #############
    print("\nтомография\n")
    radCoeffs = [[1, 1],
                 [1, 1]]

    radA = [[delta * radCoeffs[0][0], delta * radCoeffs[0][1]],
            [delta * radCoeffs[1][0], delta * radCoeffs[1][1]]]

    print("radA =\n", radA)
    A = midrad(A_orig, radA)
    print("A = \n", A)
    print(A)
    det_A = det(A)

    print(det_A)
    if m.isclose(det_A.mid, 0.0):
        print("delta = 0")
    elif 0.0 in Interval(det_A.left, det_A.right):
        min_delta = min_delta_search(A_orig, delta, radCoeffs)
        print("min delta = ", min_delta)
        radA = [[min_delta * radCoeffs[0][0], min_delta * radCoeffs[0][1]],
                [min_delta * radCoeffs[1][0], min_delta * radCoeffs[1][1]]]
        print(det(midrad(A_orig, radA)))

    ############# rad регрессии #############
    print("\nрегрессия\n")
    radCoeffs = [[1, 0],
                 [1, 0]]

    radA = [[delta * radCoeffs[0][0], delta * radCoeffs[0][1]],
            [delta * radCoeffs[1][0], delta * radCoeffs[1][1]]]

    print("radA =\n", radA)
    A = midrad(A_orig, radA)
    print("A = \n", A)
    print(A)
    det_A = det(A)

    print(det_A)
    if m.isclose(det_A.mid, 0.0):
        print("delta = 0")
    elif 0.0 in Interval(det_A.left, det_A.right):
        min_delta = min_delta_search(A_orig, delta, radCoeffs)
        print("min delta = ", min_delta)
        radA = [[min_delta * radCoeffs[0][0], min_delta * radCoeffs[0][1]],
                [min_delta * radCoeffs[1][0], min_delta * radCoeffs[1][1]]]
        # print(midrad(A_orig, radA))
        print(det(midrad(A_orig, radA)))

        # 1.05 1 0.95 1


if __name__ == "__main__":
    main()

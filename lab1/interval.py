from __future__ import annotations
import math as m
from numbers import Number
from copy import deepcopy


class Interval:
    def __init__(self, left: int | float, right: int | float) -> Interval:
        if left > right:
            raise ValueError("Left point must be less then right point")
        self.__left = left
        self.__right = right
        self.__is_point = self.__left == self.right

    def __str__(self) -> str:
        return '[' + str(self.__left) + ', ' + str(self.__right) + ']'

    def __repr__(self) -> str:
        return '[' + str(self.__left) + ', ' + str(self.__right) + ']'

    def __add__(self, other: Interval) -> Interval:
        return Interval(self.__left + other.__left, self.__right + other.__right)

    def __radd__(self, other: Interval) -> Interval:
        return Interval(self.__left + other.__left, self.__right + other.__right)

    def __sub__(self, other: Interval) -> Interval:
        return Interval(self.__left - other.__right, self.__right - other.__left)

    def __rsub__(self, other: Interval) -> Interval:
        return Interval(self.__left - other.__right, self.__right - other.__left)

    def __mul__(self, other: Interval | Number) -> Interval:
        if isinstance(other, Number):
            return Interval(self.__left * other, self.__right * other)
        if isinstance(other, Interval):
            return Interval(min(self.__left * other.__left, self.__left * other.__right,
                                self.__right * other.__left, self.__right * other.__right),
                            max(self.__left * other.__left, self.__left * other.__right,
                                self.__right * other.__left, self.__right * other.__right))
        else:
            raise TypeError("other must be Interval or Number")

    def __rmul__(self, other: Interval | Number) -> Interval:
        if isinstance(other, Number):
            return Interval(self.__left * other, self.__right * other)
        if isinstance(other, Interval):
            return Interval(min(self.__left * other.__left, self.__left * other.__right,
                                self.__right * other.__left, self.__right * other.__right),
                            max(self.__left * other.__left, self.__left * other.__right,
                                self.__right * other.__left, self.__right * other.__right))
        else:
            raise TypeError("other must be Interval or Number")

    def __truediv__(self, other: Interval) -> Interval:
        if any([m.isclose(other.__left, 0.0), m.isclose(other.__left, 0.0),
                m.isclose(other.__right, 0.0), m.isclose(other.__right, 0.0)]):
            raise ZeroDivisionError("Divide by zero.")

        return Interval(min(self.__left / other.__left, self.__left / other.__right,
                            self.__right / other.__left, self.__right / other.__right),
                        max(self.__left / other.__left, self.__left / other.__right,
                            self.__right / other.__left, self.__right / other.__right))

    def __rtruediv__(self, other: Interval) -> Interval:
        if any([m.isclose(other.__left, 0.0), m.isclose(other.__left, 0.0),
                m.isclose(other.__right, 0.0), m.isclose(other.__right, 0.0)]):
            raise ZeroDivisionError("Divide by zero.")

        return Interval(min(self.__left / other.__left, self.__left / other.__right,
                            self.__right / other.__left, self.__right / other.__right),
                        max(self.__left / other.__left, self.__left / other.__right,
                            self.__right / other.__left, self.__right / other.__right))

    def __contains__(self, num: int | float) -> bool:
        return self.__left <= num and num <= self.__right

    def is_point_interval(self) -> bool:
        return self.__is_point

    @property
    def left(self) -> int | float:
        return self.__left

    @left.setter
    def left(self, num: int | float) -> None:
        if not isinstance(num, Number):
            raise ValueError("num must be numeric.")
        self.__left = num

    @left.deleter
    def left(self) -> None:
        raise PermissionError("left is not deletable")

    @property
    def right(self) -> int | float:
        return self.__right

    @right.setter
    def right(self, num: int | float) -> None:
        if not isinstance(num, Number):
            raise ValueError("num must be numeric.")
        self.__right = num

    @right.deleter
    def right(self) -> None:
        raise PermissionError("right is not deletable")

    @property
    def mid(self) -> int | float:
        return (self.__left + self.__right) / 2

    @mid.setter
    def mid(self, *args) -> None:
        raise PermissionError("mid is not editable")

    @mid.deleter
    def mid(self) -> None:
        raise PermissionError("mid is not deletable")

    @property
    def rad(self) -> int | float:
        return (self.__right - self.__left) / 2

    @rad.setter
    def rad(self, *args) -> None:
        raise PermissionError("rad is not editable")

    @rad.deleter
    def rad(self) -> None:
        raise PermissionError("rad is not deletable")

    @property
    def wid(self) -> int | float:
        return self.__right - self.__left

    @wid.setter
    def wid(self, *args) -> None:
        raise PermissionError("wid is not editable")

    @wid.deleter
    def wid(self) -> None:
        raise PermissionError("wid is not deletable")


empty = Interval(0, 0)
infinite = Interval(-m.inf, m.inf)

IntMatrix = list[list[Interval]]
Matrix = list[list[Number]]


def midrad(midA: Matrix, radA: Matrix) -> IntMatrix:
    if len(midA) != len(radA):
        raise IndexError("The size of midA must be equal to the size of radA")
    column_count = len(midA[0])
    for i in range(len(midA)):
        if len(midA[i]) != len(radA[i]) or len(radA[i]) != column_count or len(midA[i]) != column_count:
            raise IndexError("The size of midA must be equal to the size of radA")

    res_matrix = list()
    for i in range(len(midA)):
        row = []
        for j in range(len(midA[i])):
            row.append(Interval(midA[i][j] - radA[i][j], midA[i][j] + radA[i][j]))
        res_matrix.append(row)
    return res_matrix
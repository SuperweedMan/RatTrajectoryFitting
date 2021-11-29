# %%
import sympy
import numpy as np


def intersection(a, b, tolerance=0.1):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    return b[(np.abs(a[:, None] - b) < tolerance).any(0)]


def sincos2radians(datas: np.ndarray):
    res = []
    for data in datas:
        x = sympy.symbols("x")   # 申明未知数"x"
        y = sympy.symbols("y")
        a = sympy.solve([sympy.sin(x)-data[0]], [x])   # 写入需要解的方程体
        b = sympy.solve([sympy.cos(y)-data[1]], [y])   # 写入需要解的方程体
        res.append(intersection(a, b))
    return np.array(res).astype(np.float32)


if __name__ == '__main__':
    datas = np.array([[0.99951535,  0.03113029],
                      [0.99668646,  0.08133964],
                      [0.9889875,  0.14799912],
                      [0.9618872,  0.2734467],
                      [0.9816618, -0.19063076],
                      [0.9995426, -0.03024272],
                      [0.9421902, -0.33507863],
                      [0.9322367, -0.36184916],
                      [0.9538717, -0.30021444],
                      [0.98971635,  0.14304373]])
    v = sincos2radians(datas)
# %%

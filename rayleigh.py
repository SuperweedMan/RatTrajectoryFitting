#%%
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.rayleigh(scale=13, size=1000), hist=False)

plt.show()
#%%
import sympy
x = sympy.symbols("x")   # 申明未知数"x"
a = sympy.nsolve([sympy.sin(x)-0.8062339, sympy.cos(x)- 0.59159696],[x],(0))   # 写入需要解的方程体
print (a)
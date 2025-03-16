import numpy as np

a=np.array([[1,2],
            [3,4],
            [5,6]])
#axis=0表示列，axis=1表示行
#axis为空表示全部
b=np.mean(a)
print(b)

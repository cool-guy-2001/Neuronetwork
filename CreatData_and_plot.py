#生成数据和可视化
import numpy as np
import matplotlib.pyplot as plot
import random
import math

NUM_OF_DATA=10

def tag_entry(x,y):
    dist=x**2+y**2
    if dist<1:
        tag=0
    else:
        tag=1
    return tag

def create_data(num_of_data):
    entry_list=[]
    for i in range(num_of_data):
        x=random.uniform(-2,2)
        y=random.uniform(-2,2)
        tag=tag_entry(x,y)
        entry=[x,y,tag]
        entry_list.append(entry)
    return np.array(entry_list)

#可视化
def plot_data(data,title):
    color=[]
    for i in data[:,2]:
        if i==0:
            color.append('orange')
        else:
            color.append('blue')
    plot.scatter(data[:,0],data[:,1],color=color)
    plot.title(title)
    plot.show()

if __name__ == '__main__':
    data=create_data(NUM_OF_DATA)
    print(data)
    plot_data(data,'Data')
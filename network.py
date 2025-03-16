import numpy as np
import CreatData_and_plot as cp

BATCH_SIZE=100
LEARNING_RATE=0.01
#激活函数
def activation_RELU(input):
    return np.maximum(0,input)

#标准化函数
def normalization(input):
    max_number=np.max(np.absolute(input),axis=1,keepdims=True)
    epslion=1e-8
    scale_rate=np.where(max_number==0,1,1/(max_number+epslion))
    norm=input*scale_rate
    return norm
#向量标准化函数
def vector_normalization(input):
    max_number=np.max(np.absolute(input))
    epslion=1e-8
    scale_rate=np.where(max_number==0,1,1/(max_number+epslion))
    norm=input*scale_rate
    return norm
#分类函数
def classfiy(probability):
    classification=np.rint(probability[:,1])
    return classification

#softmax激活函数
def activation_softmax(input):
    max_values =np.max(input,axis=1,keepdims=True)
    slided_input=input-max_values
    exp_values=np.exp(slided_input)
    norm_base=np.sum(exp_values,axis=1,keepdims=True)
    norm_values=exp_values/norm_base
    return norm_values

#损失函数
def precise_loss_function(predicted,real):
    real_matrix=np.zeros((len(real),2))
    print(real)
    real_matrix[:,1]=real
    real_matrix[:,0]=1-real
    product=np.sum(predicted*real_matrix,axis=1)
    return product

#需求函数
def get_final_layer_preact_demands(predicted_values,target_vetor):
    target=np.zeros((len(target_vetor),2))
    target[:,1]=target_vetor
    target[:,0]=1-target_vetor
    for i in range(len(target_vetor)):
        if np.dot(predicted_values[i],target[i])>0.5:
            target[i]=np.array([0,0])
        else :
            target[i]=(target[i]-0.5)*2
    return target


#定义一个层类 
class Layer:
    def __init__(self,inputs,outputs):
        self.weights=np.random.randn(inputs,outputs)
        self.biases=np.random.randn(outputs)
    
    def layer_output(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases
        return self.output  
    
    def layer_backward(self,preweights_values,afterweights_demands):
        preweights_demands=np.dot(afterweights_demands,self.weights.T)
        condition=(preweights_values>0)
        value_dervative=np.where(condition,1,0)
        preweights_demands=preweights_demands*value_dervative
        
        norm_preweights_demands=normalization(preweights_demands)
        weights_adjust_matrix=self.get_weights_adjust_matrix(preweights_values,afterweights_demands)
        norm_weights_adjust_matrix=normalization(weights_adjust_matrix)

        return (norm_preweights_demands,norm_weights_adjust_matrix)
    def get_weights_adjust_matrix(self,preweights_values,aftweights_demands):
        plain_weights=np.full(self.weights.shape,1)
        weights_adjust_matrix=np.full(self.weights.shape,0.0)
        plain_weights_T=plain_weights.T
        for i in range(BATCH_SIZE):
            weights_adjust_matrix+=(plain_weights_T*preweights_values[i,:]).T*aftweights_demands[i,:]
        weights_adjust_matrix=weights_adjust_matrix/BATCH_SIZE
        return weights_adjust_matrix
            
#定义一个网络类    
class Network:
    def __init__(self,network_shape):
        self.shape=network_shape
        self.layers=[]
        for i in range(len(network_shape)-1):
            layer=Layer(network_shape[i],network_shape[i+1])
            self.layers.append(layer)
    #前馈运算函数
    def network_forward(self,inputs):
        output=[inputs]     
        for i in range(len(self.layers)):
            layer_sum=self.layers[i].layer_output(output[i])
            if i==len(self.layers)-1:
                layer_output=activation_softmax(layer_sum)
            else:
                layer_output=activation_RELU(layer_sum)
                layer_output=normalization(layer_output)
            output.append(layer_output)
        return output
    
    #反向传播函数
    def network_backward(self,layer_output,targets_vector):
        backup_network=self.layers.copy()
        preAct_demands= get_final_layer_preact_demands(layer_output[-1],targets_vector)
        for i in range(len(self.layers)):
            layer=backup_network[len(self.layers)-1-i]
            if i!=0:
                layer.biases+=LEARNING_RATE*np.mean(preAct_demands,axis=0)
                layer.biases=vector_normalization(layer.biases)
            outputs=layer_output[len(layer_output)-2-i]
            results_list=layer.layer_backward(outputs,preAct_demands)
            preAct_demands=results_list[0]
            wegiths_adjust_matrix=results_list[1]
            layer.weights+=LEARNING_RATE*wegiths_adjust_matrix
            layer.weights=normalization(layer.weights)
        return backup_network
    

"""
a11=0.9
a21=-0.4

a12=-0.8
a22=0.5

a13=-0.5
a23=0.8

a14=0.7
a24=-0.3

a15=-0.9
a25=0.4

inputs=np.array([[a11,a21],
                 [a12,a22],
                 [a13,a23],
                 [a14,a24],
                 [a15,a25]])
"""



def main ():

    data=cp.create_data(BATCH_SIZE)
    cp.plot_data(data,'Right classfication')
    print(data)
    inputs=data[:,:2]
    targets=data[:,2].copy()#标准答案
    print(inputs)
    
    #建立神经网络
    network=Network([2,3,4,5,2])

    output=network.network_forward(inputs)
    classification=classfiy(output[-1])
    print(classification)
    data[:,2]=classification
    print(data)
    cp.plot_data(data,'Before Tranning')

    backup_network=network.network_backward(output,targets)
    new_output=network.network_forward(inputs)
    new_classification=classfiy(new_output[-1])
    data[:,2]=new_classification
    cp.plot_data(data,'After Tranning')
    """
    loss=precise_loss_function(output[-1],targets)
    print(loss)
    demands=get_final_layer_preact_demands(output[-1],targets)
    print(demands)
    #测试调整矩阵
    adjust_matrix=network.layers[-1].get_weights_adjust_matrix(output[-2],demands)
    print(adjust_matrix)
    #cp.plot_data(data,'Before Tranning') 
    #测试层反向传播
    layer_backwards=network.layers[-1].layer_backward(output[-2],demands)
    print(layer_backwards)
    """

    """
    第一层
    layer1=Layer(2,3)

    第二层
    layer2=Layer(3,4)
    #weights2=creat_weights(3,4)
    #biases2=creat_biases(4)

    第三层
    layer3=Layer(4,2)
    #weights3=creat_weights(4,2)
    #biases3=creat_biases(2)
    """

    """
    #第一层运算
    output1=network.layers[0].layer_output(inputs)
    print('output1') 
    print(output1)
    print('----------------')
    
    #第二层运算
    output2=network.layers[1].layer_output(output1)
    print('output2')
    print(output2)
    print('----------------') 

    #第三层运算
    output3=network.layers[2].layer_output(output2)
    print('output3')
    print(output3)
    print('----------------')
    """

"""
def test():
    network=Network([2,3,4,2])
    print(network.shape)
    print(network.layers)

test()
"""


main()

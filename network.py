import numpy as np
import CreatData_and_plot as cp
import copy
import math

BATCH_SIZE=30
LEARNING_RATE=0.01
force_train=False
random_train=False
n_improved=0
n_not_improved=0

"""
待优化的点:
1.不能自动停止，需要用肉眼判断是否接近答案
2.不能胜任复杂图样，图形一复杂起来跑的就慢了(环形图)
"""


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

#损失函数1
def precise_loss_function(predicted,real):
    real_matrix=np.zeros((len(real),2))
    #print(real)
    real_matrix[:,1]=real
    real_matrix[:,0]=1-real
    product=np.sum(predicted*real_matrix,axis=1)
    return 1-product

#损失函数2
def loss_function(predicted,real):
    condition=(predicted>0.5)
    binary_predicted=np.where(condition,1,0)
    real_matrix=np.zeros((len(real),2))
    real_matrix[:,1]=real
    real_matrix[:,0]=1-real
    product=np.sum(binary_predicted*real_matrix,axis=1)
    return 1-product


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
        backup_network=copy.deepcopy(self)
        preAct_demands= get_final_layer_preact_demands(layer_output[-1],targets_vector)
        for i in range(len(self.layers)):
            layer=backup_network.layers[len(self.layers)-1-i]
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
    
    #单批次训练
    def one_batch_train(self,batch):
        global force_train,random_train,n_improved,n_not_improved

        inputs=batch[:,(0,1)]
        targets=np.copy(batch[:,2]).astype(int)
        output=self.network_forward(inputs)
        precise_loss=precise_loss_function(output[-1],targets)
        loss=loss_function(output[-1],targets)

        if np.mean(precise_loss)<=0.1:
            print('No need for trainning')
        else:
            backup_network=self.network_backward(output,targets)
            backup_output=backup_network.network_forward(inputs)
            backup_precise_loss=precise_loss_function(backup_output[-1],targets)
            backup_loss=loss_function(backup_output[-1],targets)


            if np.mean(backup_precise_loss)<=np.mean(precise_loss)or np.mean(backup_loss)<=np.mean(loss):
                for i in range (len(self.layers)):
                    self.layers[i].weights=backup_network.layers[i].weights.copy()
                    self.layers[i].biases=backup_network.layers[i].biases.copy()
                print('Improved')
                n_improved+=1
            else:
                if force_train:
                    for i in range (len(self.layers)):
                        self.layers[i].weights=backup_network.layers[i].weights.copy()
                        self.layers[i].biases=backup_network.layers[i].biases.copy()
                    print("Force train")
                if random_train:
                    self.random_update()
                    print('Random update')
                else:
                    print('No Improvement')
                n_not_improved+=1
        print('----------------------------')

    #多批次训练
    def train(self,n_entries):
        global force_train,random_train,n_improved,n_not_improved
        n_improved=0
        n_not_improved=0

        n_batches=math.ceil(n_entries//BATCH_SIZE)
        for i in range(n_batches):
            batch=cp.create_data(BATCH_SIZE)
            self.one_batch_train(batch)
        improvement_rate=n_improved/(n_improved+n_not_improved)
        print("Improvement rate:")
        print(format(improvement_rate,".0%"))

        if improvement_rate<0.01:
            force_train=True
        else:
            force_train=False
        if n_improved==0:
            random_train=True
        else:
            random_train=False

        data=cp.create_data(800)
        #cp.plot_data(data,'Right classfication')    
        inputs=data[:,:2]
        output=self.network_forward(inputs)
        classification=classfiy(output[-1])
        data[:,2]=classification
        cp.plot_data(data,'After Tranning')
    #随机更新
    def random_update(self):
        random_network=Network([2,100,200,100,50,2])
        for i in range(len(self.layers)):
            weights_change=random_network.layers[i].weights
            biases_change=random_network.layers[i].biases
            self.layers[i].weights+=weights_change
            self.layers[i].biases+=biases_change
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

    data=cp.create_data(800)#生成数据
    cp.plot_data(data,'Right classfication')
    #print(data)
    """
    inputs=data[:,:2]
    targets=data[:,2].copy()#标准答案
    print(inputs)
    """
    #选择起始网络 
    use_this_network='n'
    while use_this_network!='Y' and use_this_network!='y':
        #建立神经网络
        #中间部分神经元数量太少，导致训练效果不太好
        #network=Network([2,3,4,5,2])
        network=Network([2,100,200,100,50,2])
        #单批次训练
        #network.one_batch_train(data)
        inputs=data[:,:2]
        output=network.network_forward(inputs)
        classification=classfiy(output[-1])
        data[:,2]=classification
        cp.plot_data(data,'chose network')
        use_this_network=input('Do you want to use this network? Y/N\n')
    #进行训练
    do_train=input('Do you want to train the network? Y/N\n')
    while do_train=='Y' or do_train=='y' or do_train.isnumeric()==True:
        if do_train.isnumeric()==True:
            n_entries=int(do_train)
        else:
            n_entries=int(input('Enter the number of entries used for training:\n'))

        network.train(n_entries)
        do_train=input('Train again? Y/N\n')
    
    #演示训练效果
    inputs=data[:,:2]
    output=network.network_forward(inputs)
    classification=classfiy(output[-1])
    data[:,2]=classification
    cp.plot_data(data,'After Tranning')
    
    #n_entries=int(input('Enter the number of entries used for training:\n'))
    #network.train(n_entries)
    """
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

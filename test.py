import numpy as np

a11=0
a21=1

a12=0.5
a22=0.5

a13=0.2
a23=0.8

a14=0.7
a24=0.3

a15=0.9
a25=0.1

predicted=np.array([[a11,a21],
                    [a12,a22],
                 [a13,a23],
                 [a14,a24],
                 [a15,a25]])

real =np.array([1,0,1,0,1])
print(predicted)

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

print(get_final_layer_preact_demands(predicted,real))
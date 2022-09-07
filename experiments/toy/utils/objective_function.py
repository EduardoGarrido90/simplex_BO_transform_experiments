import numpy as np
import torch 

def obj_fun(X_train):
    return torch.tensor([np.sin(x[0])/(np.cos(x[1]) + np.sinh(x[2])) for x in X_train])


if __name__ == '__main__' :
    X_train = torch.rand(1000,3)
    print(obj_fun(X_train))


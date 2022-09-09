import torch


def objective_function(x): #Length of x: 5. Range [0,1]^5. To be maximized.
    if torch.any(x > 1.0):
        raise Exception("Hypercube violated")

    y = torch.sum(x)
    distance_wrt_simplex = torch.abs(torch.tensor(1.0)-y)
    penalization = 0
    if distance_wrt_simplex >= 1.0:
        penalization = distance_wrt_simplex**x.shape[0] #Penalizing solutions out of diagonal more.
    else:
        penalization = distance_wrt_simplex 
    return y - penalization


if __name__ == '__main__':
    first_candidate = torch.tensor([1.0, 0.2, 0.7, 0.4, 0.5]) #They do not sum one.
    second_candidate = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]) #They sum 1.

    y_first = objective_function(first_candidate)
    y_second = objective_function(second_candidate)

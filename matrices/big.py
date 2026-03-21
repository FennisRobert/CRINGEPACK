from scipy.sparse import load_npz, csc_matrix
import numpy as np

PRE = './matrices/'

names = [('WGLight',[0,1]),]

SET_EM: list[tuple[csc_matrix, list[np.ndarray]]] = []

for name, bs in names:
    A = load_npz(PRE+name+'_A.npz')
    bs = [np.load(f'{PRE}{name}_b{i}.npy') for i in bs]
    SET_EM.append((A, bs))    


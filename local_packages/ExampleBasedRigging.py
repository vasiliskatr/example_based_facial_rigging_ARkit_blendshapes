import numpy as np
import os
import time
import local_packages.tools3d_ as t3d 
from scipy.optimize import lsq_linear
from IPython.display import clear_output
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from qpsolvers import solve_qp
import numba
from numba import jit
from numba.core.extending import overload
from numba.np.linalg import norm_impl
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
from tqdm import tqdm



def reading_generic_bs (objpath_generic_bs):
    A_bs_model = [] # variable to store all data with regards to the template 
    B_bs_model = [] # variable to store all data with regards to the actor
    bs_name_list = [] # variable to store the names of imported BS
    A_0, faces, _, _ = t3d.Read(objpath_generic_bs + 'Neutral.obj', QuadMode = True)
    n_vertices = A_0.shape[1]
    #A_0 = tools_3d.center(A_0)
    generic_bs_data = os.scandir(objpath_generic_bs)
    for generic_bs in tqdm(generic_bs_data, unit=' files', desc="Loading generic blend shapes"):
        name, ext = os.path.splitext(generic_bs)
        name_s = name.split("/")
       
        if ext == '.obj' and 'neutral' not in name:  # read only the .obj files from the source directory
            temp_vertices, _, _, _ = t3d.Read(name+ext, QuadMode = True)
            A_bs_vertices = temp_vertices - A_0
            A_bs_model.append(A_bs_vertices)
            B_bs_model.append(np.zeros((3, n_vertices)))
            bs_name_list.append(name_s[-1])
            
    n = len(A_bs_model)
    print ('Generic model of n = ' + str(len(A_bs_model)) + ' blend shapes imported (+1 neutral pose)') 
    return A_bs_model, B_bs_model, A_0, faces, n, bs_name_list



def reading_training_data(objpath_training_poses):    
    S_training_poses = [] # Variable to store training poses
    training_pose_data = os.scandir(objpath_training_poses)

    for training_pose in tqdm(training_pose_data, unit=' files', desc="Loading poses"):
        name, ext = os.path.splitext(training_pose)

        if ext == '.obj':  # read only the .obj files from the source directory    
                temp_vertices, _, _, _ = t3d.Read(name+ext, QuadMode = True)
                
                S_training_poses.append(temp_vertices)
                
    m = len(S_training_poses)  
    print ('m = ' + str(m)+' training poses in total (+ 1 neutral)')
    return S_training_poses, m



def blend_shape_weights(A_0, B_0, A_BS_model, S_training_poses):
    # initial blend shape weights guessing for each training pose

    start_time = time.time()
    print ('Computing intial blend-shape weight guess for the training poses')
    n = len(A_BS_model)
    m = len(S_training_poses)
    n_vertices = A_0.shape[1]
    
    Alpha_star = np.zeros((m, n))# Initial guess of blendhspae weights

    A_All = A_BS_model[0].T.flatten().reshape(n_vertices*3, 1)

    for i in  range(1,n):
        A_All = np.concatenate((A_All, A_BS_model[i].T.flatten().reshape(n_vertices*3, 1)), axis=1)

    for i in tqdm(range(m), unit=' pose', desc='Guessing weights'):        
        B_temp = (S_training_poses[i] - B_0).T.flatten().reshape(n_vertices*3)
        weights_temp = lsq_linear(A_All, B_temp, bounds = (0, 1), lsmr_tol='auto', verbose=0)
        Alpha_star[i, :] = weights_temp.x.reshape(1, n)
    
    return Alpha_star


    print ("done in ",(time.time() - start_time), "sec")      


def columnise(model):
    for i in range(0, len(model)):
        model[i] = model[i].T

    return model


@jit(nopython=True, parallel=True)
def local_tri_frame_fast(vertices, triangles, tri_index):
    
    tri_vertices = vertices[triangles[tri_index, :], :]
    
    LF = np.zeros((3,3))
    
    v1 = tri_vertices[0, :]
    v2 = tri_vertices[1, :]
    v3 = tri_vertices[2, :]
    
    LF[:,0] = (v3-v1) # v3-v1
    LF[:,1] = (v2-v1) # v2-v1
    LF[:,2] = (np.cross((v3-v1),(v2-v1))) # n
    
    return LF


@jit(nopython=True, parallel=True)
def compute_lf_fast (vertices, triangles):

    lf = np.zeros((len(triangles)*3, 3))
    for i in numba.prange(len(triangles)): 
        lf[i*3:i*3+3]= local_tri_frame_fast(vertices, triangles, i)
    
    return lf


@jit(nopython=True, parallel=True)
def compute_lf_inverse_fast(vertices, triangles):
  
    lf_inv = np.zeros((len(triangles)*3, 3))
    for i in numba.prange(len(triangles)):  
        lf_inv[i*3:i*3+3] = np.linalg.inv(local_tri_frame_fast(vertices, triangles, i))
    return lf_inv


@jit(nopython=True, parallel=True)
def make_M_S_minus_M_B_0_fast(S_training_poses, B_0, triangles):
    
    m = len(S_training_poses)
    M_B_0 = compute_lf_fast(B_0, triangles)    
    M_S_minus_M_B_0 = np.empty((m, len(triangles)*3, 3))
    M_S = np.empty((m, len(triangles)*3, 3))

    for s in numba.prange(m):
        M_S_temp = compute_lf_fast(S_training_poses[s], triangles)
        M_S_minus_M_B_0[s] = M_S_temp - M_B_0     
        M_S[s] = M_S_temp
      
    return M_S_minus_M_B_0 , M_B_0, M_S


@jit(nopython=True, parallel=True)
def make_W_seed_fast(triangles, A_BS_model, kappa, theta):  
    n = len(A_BS_model) 
    W_seed = np.empty((n, len(triangles)))
    for i in numba.prange(n):
        M_A_i = compute_lf_fast(A_BS_model[i], triangles)
        
        for j in numba.prange(len(triangles)):
            lf_tri_norm = np.linalg.norm(M_A_i[j*3:j*3+3,:])
            W_seed[i,j] = (1 + lf_tri_norm)/np.power((kappa + lf_tri_norm), theta)
            
    return W_seed


@jit(nopython=True, parallel=True)
def make_M_A_star_fast(triangles, A_0, B_0, A_BS_model):
    n = len(A_BS_model)
    M_A_star = np.empty((n, len(triangles)*3, 3))
    M_A_0_inv = compute_lf_inverse_fast(A_0, triangles) 
    M_A_0 = compute_lf_fast(A_0, triangles)
    M_B_0 = compute_lf_fast(B_0, triangles)

    for i in numba.prange(n):
        M_A_i = compute_lf_fast(A_BS_model[i], triangles)
        M_A_sum = M_A_0 + M_A_i 
        
        for j in numba.prange(len(triangles)):   
            M_A_star[i][j*3:j*3+3] = ((M_A_sum[j*3:j*3+3] @ M_A_0_inv[j*3:j*3+3]) @ M_B_0[j*3:j*3+3]) - M_B_0[j*3:j*3+3]
        
    return M_A_star
    

    
# Parallel version lf optimisation
@jit(nopython=True, parallel=True)
def lf_optimisation (num_triangles, A, M_S_minus_M_B_0, M_B, M_A_star, beta, gamma, W_seed, opt_iteration, n, m):
    
    for tri_index in numba.prange(num_triangles): 
         # Constructing Bfit
        B_fit = np.zeros((n*3,3))
        B_fit = A.T @ M_S_minus_M_B_0[:,tri_index*3:tri_index*3+3,:].copy().reshape(m*3,3)
         
        # Constructing W   
        dia = [[i,i,i] for i in W_seed[:,tri_index]]
        dia = np.asarray(dia)
        dia = dia.flatten()
        W = np.diag(dia, 0)
        M_A_starr = M_A_star[:,tri_index*3:tri_index*3+3,:].copy().reshape(n*3,3) 
        A_sum = A.T @ A + beta[opt_iteration] * (W.T @ W)
        B_sum = B_fit + beta[opt_iteration] * (W.T @ (W @ M_A_starr))                 
        M_B_tri = np.linalg.solve(A_sum, B_sum[:,0:2])    #.copy() 
        M_B[:, tri_index*2:tri_index*2+2] = M_B_tri #.copy()
        
    return  M_B 

def make_A_sparse_reconstruction (triangles, n_vertices):
    row = []
    col = []
    data = []
        
    for j in range(len(triangles)):
        tri_indices = triangles[j]

        row.append(j*2)
        col.append(tri_indices[2])
        data.append(1)

        row.append(j*2)
        col.append(tri_indices[0])
        data.append(-1)

        row.append(j*2+1)
        col.append(tri_indices[1])
        data.append(1)

        row.append(j*2+1)
        col.append(tri_indices[0])
        data.append(-1)
        
    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)
        
    ########### I removed the consideration of zero-deformation vertices.
    ########### There is no drifting in the reconstruction even without it.    
    A_sparse = csr_matrix((data, (row, col)), shape=(triangles.shape[0]*2, n_vertices))
    return A_sparse


def recon(M_B, A_sparse_recon, n_vertices, num_triangles, i):
      # reconstruction of vertices 
    
    B_temp_X = M_B[i*3,:].reshape(num_triangles*2, 1)
    X_vals = sp.linalg.lsqr(A_sparse_recon, B_temp_X)[0]
    B_temp_Y = M_B[i*3+1,:].reshape(num_triangles*2, 1)
    Y_vals = sp.linalg.lsqr(A_sparse_recon, B_temp_Y)[0]
    B_temp_Z = M_B[i*3+2,:].reshape(num_triangles*2, 1)
    Z_vals = sp.linalg.lsqr(A_sparse_recon, B_temp_Z)[0]
    
    return X_vals.reshape(1, n_vertices), Y_vals.reshape(1, n_vertices), Z_vals.reshape(1, n_vertices), i 




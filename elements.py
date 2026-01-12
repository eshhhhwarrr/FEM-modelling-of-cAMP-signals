import time

start_time = time.time()



import numpy as np
from mpi4py import MPI
import meshio
import gmsh
import numpy as np

gmsh.initialize()
gmsh.model.add("triangles")

lc = 0.1
p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
p2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
p3 = gmsh.model.geo.addPoint(1, 1, 0, lc)
p4 = gmsh.model.geo.addPoint(0, 1, 0, lc)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surface = gmsh.model.geo.addPlaneSurface([loop])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

gmsh.write("triangles.msh")

gmsh.finalize()

import meshio
import numpy as np

mesh = meshio.read("triangles.msh")

points = mesh.points  # x, y, z
indices=np.zeros([142,3])

for i in range(len(points)):
    x,y,z=points[i]
    indices[i][2]=i+1
    indices[i][0]=points[i][0]
    indices[i][1]=points[i][1]

points=points[:,:2]
print(np.shape(points))

cells = mesh.cells_dict["triangle"]  

triangle_coords = points[cells]  

print(np.shape(triangle_coords))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

E = 1  
nu = 0  
t = 0.01   

D = (E / (1 - nu**2)) * np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1 - nu) / 2]
])

print(indices)
# print((points))
# print(triangle_coords)

def elements(triangle_coords):
    n_triangles = triangle_coords.shape[0]
    result = []
    for tri in triangle_coords:
        one = np.ones((3, 1))
        element = np.hstack((one, tri))
        result.append(element)
    return np.array(result)

# print(elements(triangle_coords))


def gradient_shape_function(element,triangle_coords):
    for i in range(len(triangle_coords[0])):
        x1, y1 = element[i,0, 1], element[i,0, 2]
        x2, y2 = element[i,1, 1], element[i,1, 2]
        x3, y3 = element[i,2, 1], element[i,2, 2]
    A = 0.5 
    dNdx = np.array([(y2 - y3), (y3 - y1), (y1 - y2)]) / (2 * A)
    dNdy = np.array([(x3 - x2), (x1 - x3), (x2 - x1)]) / (2 * A)
    gradN = np.vstack((dNdx, dNdy))
    return gradN
print(gradient_shape_function(elements(triangle_coords),triangle_coords))

def strain_displacement_matrix(gradN, npoint):
    B = np.zeros((3, 2 * len(npoint)))
    for i in range(len(npoint)):
        B[0, 2 * i]     = gradN[0, i]
        B[1, 2 * i + 1] = gradN[1, i]
        B[2, 2 * i]     = gradN[1, i]
        B[2, 2 * i + 1] = gradN[0, i]
    return B

def stiffness_matrix(element, npoint):
    gradN, A = gradient_shape_function(element)
    B = strain_displacement_matrix(gradN, npoint)
    return A * (B.T @ D @ B)

# def assemble_stiffness(K_local, indices, K_global):
#     for i in range(len(indices)):
#         for j in range(len(indices)):
#             K_global[indices[i], indices[j]] += K_local[i, j]
#     return K_global

# if size == 1:
#     K_global = np.zeros((10, 10))

#     for i in range(len(triangles)):
#         npoint = triangles[i]
#         indices = indices_list[i]
#         element = elements(npoint)
#         K_local = stiffness_matrix(element, npoint)
#         assemble_stiffness(K_local, indices, K_global)

# else:
#     # Flexible parallel mode: distribute triangles across available ranks
#     local_K_list = []

#     # Each rank takes every `size`-th triangle starting from its own rank
#     for i in range(rank, len(triangles), size):
#         npoint = triangles[i]
#         indices = indices_list[i]
#         element = elements(npoint)
#         K_local = stiffness_matrix(element, npoint)
#         local_K_list.append((K_local, indices))

#     # Gather all element contributions from all ranks
#     all_data = comm.gather(local_K_list, root=0)

#     if rank == 0:
#         K_global = np.zeros((10, 10))
#         for worker_data in all_data:
#             for K_elem, indices in worker_data:
#                 assemble_stiffness(K_elem, indices, K_global)
# # else:
# #     local_npoint = triangles[rank]
# #     local_indices = indices_list[rank]
# #     element = elements(local_npoint)
# #     local_K = stiffness_matrix(element, local_npoint)
# #     local_data = (local_K, local_indices)
    
# #     all_data = comm.gather(local_data, root=0)
# #     # local_K_list = []
# #     # local_indices_list = []

# #     # for i in range(rank, len(triangles), size):
# #     #     npoint = triangles[i]
# #     #     indices = indices_list[i]
# #     #     element = elements(npoint)
# #     #     K_local = stiffness_matrix(element, npoint)
# #     #     local_K_list.append((K_local, indices))

# #     # all_data = comm.gather(local_K_list, root=0)

# #     # if rank == 0:
# #     #     K_global = np.zeros((10, 10))
# #     #     for K_elem, indices in all_data:
# #     #         assemble_stiffness(K_elem, indices, K_global)



    
# u = np.zeros(10)
# f = np.zeros(10)
# u[2] = 1
# u[4] = 1

# free_dofs = [3, 5, 8, 9]
# fixed_dofs = [0, 1, 2, 4, 6, 7]

# k_ff = K_global[np.ix_(free_dofs, free_dofs)]
# k_fc = K_global[np.ix_(free_dofs, fixed_dofs)]
# f_f = f[free_dofs]
# u_c = u[fixed_dofs]

# u_f = np.linalg.solve(k_ff, f_f - k_fc @ u_c)
# u[free_dofs] = u_f

# print("Final displacement vector u:")
# print(u)


# print("--- %s seconds ---" % (time.time() - start_time))
import gmsh
import math
import os

def generate_annular_mesh(filename="annular.msh", r_in=5, r_out=10):
    for lc in [1.0]:
        gmsh.initialize()
        gmsh.model.add("annular")
        center = gmsh.model.geo.addPoint(0, 0, 0, lc)  
        outer = []
        for i in range(4):
            angle = i * math.pi / 2
            p = gmsh.model.geo.addPoint(r_out * math.cos(angle), r_out * math.sin(angle), 0, lc)
            outer.append(p)
        outer_arcs = []
        for i in range(4):
            arc = gmsh.model.geo.addCircleArc(outer[i], center, outer[(i + 1) % 4])
            outer_arcs.append(arc)
        inner = []
        for i in range(4):
            angle = i * math.pi / 2
            p = gmsh.model.geo.addPoint(r_in * math.cos(angle), r_in * math.sin(angle), 0, lc)
            inner.append(p)
        inner_arcs = []
        for i in range(4):
            arc = gmsh.model.geo.addCircleArc(inner[i], center, inner[(i + 1) % 4])
            inner_arcs.append(arc)

        cl_outer = gmsh.model.geo.addCurveLoop(outer_arcs)
        cl_inner = gmsh.model.geo.addCurveLoop(inner_arcs)
        surf = gmsh.model.geo.addPlaneSurface([cl_outer, cl_inner])

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(filename)
        gmsh.finalize()
        print(f"Mesh saved to {filename}")



if __name__ == "__main__":
    generate_annular_mesh()

import numpy as np
import meshio
import matplotlib.pyplot as plt
import pandas as pd
import os

def read_mesh(filename="annular.msh"):
    mesh = meshio.read(filename)
    points = mesh.points[:, :2]
    elements = mesh.cells_dict["triangle"]
    print("Number of nodes:", len(points))
    print("Number of elements:", len(elements))
    print("Number of unique nodes:", len(np.unique(points, axis=0)))
    return points, elements

def local_matrices(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    A = 0.5 * np.linalg.det(np.array([[1, x[0], y[0]],
                                      [1, x[1], y[1]],
                                      [1, x[2], y[2]]]))
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    B = np.array([b, c]) / (2*A)
    Ke = A * (B.T @ B)
    Me = (A / 12) * (np.ones((3, 3)) + np.eye(3))
    return Ke, Me

def get_boundary_conditions(nodes, r_outer=10.0, r_inner=5.0):
    bc_nodes = []
    C_bc = {}
    r_vals = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    print("r range in mesh:", r_vals.min(), "to", r_vals.max())
    
    for i, r in enumerate(r_vals):
        if abs(r - r_outer) < 0.1:
            bc_nodes.append(i)
            C_bc[i] = 3  
    
    for i,r in enumerate(r_vals):
        if abs(r-r_inner)<0.1:
            bc_nodes.append(i)
            C_bc[i] = 1

    center_index = np.argmin(r_vals)
    bc_nodes.append(center_index)
    C_bc[center_index] = 0.0

    print(f"Dirichlet BC applied to {len(bc_nodes)} nodes (including center)")
    return bc_nodes, C_bc

def apply_dirichlet(K, f, bc_nodes, C_bc):
    for i in bc_nodes:
        K[i, :] = 0
        K[:, i] = 0
        K[i, i] = 1.0
        f[:] -= C_bc[i] * K[:, i]  
        f[i] = C_bc[i]
    return K, f

# def degradation(C, Vmax=0.295, Km=2.6):
#     return (Vmax * C) / (Km + C)

# def generate_EAC(nodes, r_outer=10.0, value=15):
#     EAC_vec = np.zeros(len(nodes))
#     r_vals = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
#     for i, r in enumerate(r_vals):
#         if abs(r - r_outer) < 0.5:
#             EAC_vec[i] = value
#     return EAC_vec


def plot_concentration(nodes, elements, C, time):
    plt.figure(figsize=(6, 5))
    plt.tricontourf(nodes[:, 0], nodes[:, 1], elements, C, levels=30, cmap='plasma')
    plt.title(f"C(x, y, t={time:.3f} s)")
    plt.colorbar(label="Concentration")
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"contour_t{time:.3f}.png")
    plt.close()

def run_simulation(nodes, elements, dt=0.01, total_time=4, D=30.0, ts=0, td=0):
    n = len(nodes)
    K = np.zeros((n, n))
    M = np.zeros((n, n))

    for tri in elements:
        xy = nodes[tri]
        Ke, Me = local_matrices(xy)
        for i in range(3):
            for j in range(3):
                K[tri[i], tri[j]] += D * Ke[i, j]
                M[tri[i], tri[j]] += Me[i, j]

    A = M + 0.5 * dt * K
    B = M - 0.5 * dt * K

    # C = np.zeros(n)
    # center_index = np.argmin(np.linalg.norm(nodes, axis=1))  # closest to (0, 0)
    # C = np.zeros(n)
    C = np.full(n, 0.1)  # Initial condition (μM)
    # EAC_vec = generate_EAC(nodes)
    node_index = np.argmin(np.abs(np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2) - 8.267))  # node between boundaries


    bc_nodes, C_bc = get_boundary_conditions(nodes)
    if len(bc_nodes) == 0:
        raise ValueError("No Dirichlet boundary nodes found. Matrix will be singular.")

    times = []
    center_concs = []

    for step in range(int(total_time / dt) + 1):
        t = step * dt
        rhs = B @ C
        # if t<ts:
        #     rhs+=dt
        A_bc, rhs_bc = apply_dirichlet(A.copy(), rhs.copy(), bc_nodes, C_bc)

        if np.linalg.matrix_rank(A_bc) < len(A_bc):
            print("WARNING: Matrix is still singular. Rank deficiency:", len(A_bc) - np.linalg.matrix_rank(A_bc))
            raise np.linalg.LinAlgError("A_bc is singular before solve.")

        C = np.linalg.solve(A_bc, rhs_bc)

        if np.isclose(t, [0.001, 0.01, 0.1, 0.5], atol=dt/2).any():
            plot_concentration(nodes, elements, C, t)
        # elif t >= ts and t < td:
        #     rhs += dt * EAC_vec
        # elif t >= td:
        #     rhs += dt * (EAC_vec - degradation(C))

        # A_bc, rhs_bc = apply_dirichlet(A.copy(), rhs.copy(), bc_nodes, C_bc)

        # if np.linalg.matrix_rank(A_bc) < len(A_bc):
        #     print("WARNING: Matrix is still singular. Rank deficiency:", len(A_bc) - np.linalg.matrix_rank(A_bc))
        #     raise np.linalg.LinAlgError("A_bc is singular before solve.")

        # C = np.linalg.solve(A_bc, rhs_bc)

        # if np.isclose(t, [0.1,1, 50, 100, 150.0,200], atol=dt/2).any():
        #     plot_concentration(nodes, elements, C, t)
        times.append(t)
        center_concs.append(C[node_index])

    df = pd.DataFrame({
        "Time (s)": times,
        "Concentration": center_concs
    })
    df.to_csv("center_concentration.csv", index=False)
    print("Saved center_concentration.csv")

    return times, center_concs

if __name__ == "__main__":
    if not os.path.exists("annular.msh"):
        print("Mesh file annular.msh not found. Run the mesh generator first.")
        exit()

    nodes, elements = read_mesh("annular.msh")
    times, center_concs = run_simulation(nodes, elements)

    plt.figure()
    plt.plot(times, center_concs, label="C(center)", color="darkred")
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration")
    plt.title("cAMP Concentration vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("center_vs_time.png")
    plt.show()


import numpy as np

# Regex for replacing jnp's A = A.at[foo].set(bar) with np's A[foo] = bar
# Find:  = (.*).at\[(.*)\].set\((.*)\)
# Replace: [$2] = $3

# Generate a adjacency matrix for an NxN rectangular network.
# Nodes are labeled, for example (N=4):
#                                           row
#
#            11    10     9     8           5
#            |      |     |     |
#      12 -- 28 -- 29 -- 30 -- 31 -- 7      4
#            |      |     |     |
#      13 -- 24 -- 25 -- 26 -- 27 -- 6      3
#            |      |     |     |
#      14 -- 20 -- 21 -- 22 -- 23 -- 5      2
#            |      |     |     |
#      15 -- 16 -- 17 -- 18 -- 19 -- 4      1
#            |      |     |     |
#            0      1     2     3           0
#
#
# col  0     1      2     3     4    5
#
#
# TODO: So much of this code is dealing with these rectangular graphs.
# This code could be abstracted and reused, maybe for simulations.


def num_edges(N):
    return (N + 1) * N * 2


def num_nodes(N):
    return 4 * N + N**2


# TODO: rename to just "index"
def interior_index(N, row, col):
    if row == 0:  # South
        return col - 1
    if col == N + 1:  # East
        return N + row - 1
    if row == N + 1:  # North
        return 3 * N - col
    if col == 0:  # West
        return 4 * N - row
    return (4 * N - 1) + N * (row - 1) + col


def node_to_row_col(N, node):
    if node < N:  # South
        return (0, node + 1)
    if node in range(N, 2 * N):  # East
        return (node - N + 1, N + 1)
    if node in range(2 * N, 3 * N):  # North
        return (N + 1, N - (node - 2 * N))
    if node in range(3 * N, 4 * N):  # West
        return (N - (node - 3 * N), 0)
    return (node - 4 * N) // N + 1, (node - 4 * N) % N + 1


def neighbors(N, node):
    row, col = node_to_row_col(N, node)
    if node < N:  # South
        return [interior_index(N, row + 1, col)]
    if node in range(N, 2 * N):  # East
        return [interior_index(N, row, col - 1)]
    if node in range(2 * N, 3 * N):  # North
        return [interior_index(N, row - 1, col)]
    if node in range(3 * N, 4 * N):  # West
        return [interior_index(N, row, col + 1)]
    return [
        interior_index(N, row - 1, col),
        interior_index(N, row, col - 1),
        interior_index(N, row, col + 1),
        interior_index(N, row + 1, col),
    ]


def K_dim(N):
    return num_nodes(N)


def boundary_slice_S(N):
    return slice(0 * N, 1 * N)


def boundary_slice_E(N):
    return slice(1 * N, 2 * N)


def boundary_slice_N(N):
    return slice(2 * N, 3 * N)


def boundary_slice_W(N):
    return slice(3 * N, 4 * N)


def build_adjacency_matrix(N):
    A = np.zeros([K_dim(N), K_dim(N)])

    # Build upper-triangular part of A node-by-node.
    # Start with boundary nodes.
    # South edge
    row = 0
    for col in range(1, N + 1):
        boundary_node = col - 1
        interior_node = interior_index(N, row + 1, col)
        A[boundary_node, interior_node] = 1.0
    # East edge
    col = N + 1
    for row in range(1, N + 1):
        boundary_node = N + row - 1
        interior_node = interior_index(N, row, col - 1)
        A[boundary_node, interior_node] = 1.0
    # North edge
    row = N + 1
    for col in range(N, 0, -1):
        boundary_node = 2 * N + (N - col)
        interior_node = interior_index(N, row - 1, col)
        A[boundary_node, interior_node] = 1.0
    # West edge
    col = 0
    for row in range(N, 0, -1):
        boundary_node = 3 * N + (N - row)
        interior_node = interior_index(N, row, col + 1)
        A[boundary_node, interior_node] = 1.0

    # Interior nodes
    # At every node, edge connections to the W and S have already been calculated.
    # At col N, edge connections to the E have already been calculated.
    # At row N, edge connections to the N have already been calculated.
    for row in range(1, N):
        for col in range(1, N):
            node = interior_index(N, row, col)
            E_node = interior_index(N, row, col + 1)
            A[node, E_node] = 1.0
            N_node = interior_index(N, row + 1, col)
            A[node, N_node] = 1.0
        col = N
        node = interior_index(N, row, col)
        N_node = interior_index(N, row + 1, col)
        A[node, N_node] = 1.0
    row = N
    for col in range(1, N):
        node = interior_index(N, row, col)
        E_node = interior_index(N, row, col + 1)
        A[node, E_node] = 1.0

    # Make K symmetric
    A = A + A.T

    return A


def build_kirchoff_matrix_from_adjacency_mask(Y, adj):
    """
    Y is a matrix of conductivities with the same shape as A.
    adj is an adjacency matrix.
    Mask Y using A, then fill in the diagonal entries to make a Kirchoff matrix.
    """
    Y = (Y + Y.T) / 2  # Ensure Y is symmetric
    K = Y * adj
    K = K - np.diag(np.sum(K, axis=1))  # Fill in diagonal entries
    return K


def decompose_kirchoff_matrix(N, K):
    """
        K = [ A    B ]
            [ B^T  D ]
    Returns A, B, D.
    """
    i = 4 * N  # index of first interior node
    A = K[:i, :i]
    B = K[:i, i:]
    D = K[i:, i:]
    return A, B, D


# The forward operator is a Schur complement of the Kirchoff matrix
#   \Lambda = A - B D^{-1} B^T
def forward(N, K):
    A, B, D = decompose_kirchoff_matrix(N, K)
    return A - B @ np.linalg.solve(D, B.T)


# Flatten a Kirchoff matrix to a **column vector** (size num_edges(N) \times 1) of edge conductivities
def flatten_K(N, K, adj):
    flat_K = np.empty([num_edges(N), 1])
    k = 0
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] == 1:
                flat_K[k, 0] = K[i, j]
                k = k + 1
    return flat_K


def unflatten_K(N, flat_K, adj):
    K = np.zeros([K_dim(N), K_dim(N)])
    k = 0
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] == 1:
                K[i, j] = flat_K[k, 0]
                k = k + 1
    K = K + K.T  # Make Symmetric
    K = K - np.diag(np.sum(K, axis=1))  # Fill in diagonal entries
    return K


# assert (KK[0] == unflatten_K(N, flatten_K(N, KK[0], adj), adj)).all()

# Rotate a network?
# def rotate(N?):
#   pass


def curtis_morrow_bak(N, Lambda):
    K = np.zeros([K_dim(N), K_dim(N)])

    S_idx = slice(0 * N, 1 * N)
    E_idx = slice(1 * N, 2 * N)
    N_idx = slice(2 * N, 3 * N)
    W_idx = slice(3 * N, 4 * N)

    # Potential and current functions, indexed by the boundary node we set to 1.0
    u = np.zeros([num_edges(N), K_dim(N)])
    c = np.zeros([num_edges(N), K_dim(N), K_dim(N)])

    def update_u_c(u, c, edge_range, zero_c_edge, opp_c_edge):
        for i in edge_range:
            y = np.zeros(4 * N)
            y[i] = 1.0
            psi = Lambda @ y
            x_opp = np.linalg.solve(Lambda[zero_c_edge, opp_c_edge], -psi[zero_c_edge])
            x = np.zeros(4 * N)
            x[opp_c_edge] = x_opp
            x[i] = 1.0
            phi = Lambda @ x

            u[i, : 4 * N] = x

            for j in range(N):  # South
                b_idx = j  # boundary index
                i_idx = interior_index(N, 1, j + 1)
                c[i, b_idx, i_idx] = phi[b_idx]
            for j in range(N):  # East
                b_idx = N + j  # boundary index
                i_idx = interior_index(N, j + 1, N)
                c[i, b_idx, i_idx] = phi[b_idx]
            for j in range(N):  # North
                b_idx = 2 * N + j  # boundary index
                i_idx = interior_index(N, N, N - j)
                c[i, b_idx, i_idx] = phi[b_idx]
            for j in range(N):  # West
                b_idx = 3 * N + j  # boundary index
                i_idx = interior_index(N, N - j, 1)
                c[i, b_idx, i_idx] = phi[b_idx]

        return u, c

    u, c = update_u_c(u, c, range(0 * N, 1 * N), E_idx, W_idx)  # South
    u, c = update_u_c(u, c, range(1 * N, 2 * N), S_idx, N_idx)  # East
    u, c = update_u_c(u, c, range(2 * N, 3 * N), W_idx, E_idx)  # North
    u, c = update_u_c(u, c, range(3 * N, 4 * N), N_idx, S_idx)  # West

    # Normal shell 0 (Boundary spikes)
    # South
    for i, q in enumerate(range(0, N)):
        z = interior_index(N, 1, 1 + i)  # interior node
        K[q, z] = (
            Lambda[q, W_idx] @ np.linalg.solve(Lambda[E_idx, W_idx], Lambda[E_idx, q])
            - Lambda[q, q]
        )
    # East
    for i, q in enumerate(range(1 * N, 2 * N)):
        z = interior_index(N, 1 + i, N)  # interior node
        K[q, z] = (
            Lambda[q, S_idx] @ np.linalg.solve(Lambda[N_idx, S_idx], Lambda[N_idx, q])
            - Lambda[q, q]
        )
    # North
    for i, q in enumerate(range(2 * N, 3 * N)):
        z = interior_index(N, N, N - i)  # interior node
        K[q, z] = (
            Lambda[q, E_idx] @ np.linalg.solve(Lambda[W_idx, E_idx], Lambda[W_idx, q])
            - Lambda[q, q]
        )
    # West
    for i, q in enumerate(range(3 * N, 4 * N)):
        z = interior_index(N, N - i, 1)  # interior node
        K[q, z] = (
            Lambda[q, N_idx] @ np.linalg.solve(Lambda[S_idx, N_idx], Lambda[S_idx, q])
            - Lambda[q, q]
        )

    # Tangent shell 1

    # South potential spike at q2
    #
    #
    # r1 -- r2
    #  |     |
    # q1    q2
    row = 1
    for i, col in enumerate(range(1, N)):
        q1 = i
        q2 = i + 1
        r1 = interior_index(N, row, col)
        r2 = interior_index(N, row, col + 1)

        u[q2, r1] = -c[q2, q1, r1] / K[q1, r1]
        K[r2, r1] = c[q2, q2, r2] / -u[q2, r1]

    # East potential spike at q2
    #
    # r1 -- q1
    # |
    # r2 -- q2
    col = N
    for i, row in enumerate(range(N, 1, -1)):
        q1 = 2 * N - i - 1
        q2 = q1 - 1
        r1 = interior_index(N, row, col)
        r2 = interior_index(N, row - 1, col)

        u[q2, r1] = -c[q2, q1, r1] / K[q1, r1]
        K[r2, r1] = c[q2, q2, r2] / -u[q2, r1]

    # North potential spike at q2
    #
    # q2    q1  (Boundary)
    #  |     |
    # r2 -- r1  (Interior)
    row = N
    for i, col in enumerate(range(N, 1, -1)):
        q1 = 2 * N + i
        q2 = q1 + 1
        r1 = interior_index(N, row, col)
        r2 = interior_index(N, row, col - 1)

        u[q2, r1] = -c[q2, q1, r1] / K[q1, r1]
        K[r2, r1] = c[q2, q2, r2] / -u[q2, r1]

    # West potential spike at q2
    #
    # q2 -- r2
    #        |
    # q1 -- r1
    col = 1
    for i, row in enumerate(range(1, N)):
        q1 = 4 * N - i - 1
        q2 = q1 - 1
        r1 = interior_index(N, row, col)
        r2 = interior_index(N, row + 1, col)

        u[q2, r1] = -c[q2, q1, r1] / K[q1, r1]
        K[r2, r1] = c[q2, q2, r2] / -u[q2, r1]

    if N == 2:
        K = K + K.T  # Make Symmetric
        K = K - np.diag(np.sum(K, axis=1))  # Fill in diagonal entries
        return K

    # Normal shell 1

    # South potential spike at q3
    #
    #       t2
    #        |
    # r1 -- s1 -- t1
    #  |     |     |
    # q1    q2    q3
    #
    row = 1
    for i, col in enumerate(range(1, N - 1)):
        q1 = i
        q2 = q1 + 1
        q3 = q2 + 1
        r1 = interior_index(N, row, col)
        s1 = interior_index(N, row, col + 1)
        t1 = interior_index(N, row, col + 2)
        t2 = interior_index(N, row + 1, col + 1)

        # Calculate c[q3,t2,s1] using Kirchoff's Law
        # Already have u[q3,s1]
        u[q3, r1] = -c[q3, q1, r1] / K[q1, r1]
        c[q3, s1, r1] = K[s1, r1] * (u[q3, s1] - u[q3, r1])  # Ohm's Law
        c[q3, t1, s1] = -c[q3, q3, t1]
        c[q3, t2, s1] = -c[q3, q2, s1] + c[q3, t1, s1] + c[q3, s1, r1]
        K[t2, s1] = -c[q3, t2, s1] / u[q3, s1]

    # East potential spike at q3
    #
    #       r1 -- q1
    #        |
    # t2 -- s1 -- q2
    #        |
    #       t1 -- q3
    col = N
    for i, row in enumerate(range(N, 2, -1)):
        q1 = 2 * N - i - 1
        q2 = q1 - 1
        q3 = q2 - 1
        r1 = interior_index(N, row, col)
        s1 = interior_index(N, row - 1, col)
        t1 = interior_index(N, row - 2, col)
        t2 = interior_index(N, row - 1, col - 1)

        # Calculate c[q3,t2,s1] using Kirchoff's Law
        # Already have u[q3,s1]
        u[q3, r1] = -c[q3, q1, r1] / K[q1, r1]
        c[q3, s1, r1] = K[s1, r1] * (u[q3, s1] - u[q3, r1])  # Ohm's Law
        c[q3, t1, s1] = -c[q3, q3, t1]
        c[q3, t2, s1] = -c[q3, q2, s1] + c[q3, t1, s1] + c[q3, s1, r1]
        K[t2, s1] = -c[q3, t2, s1] / u[q3, s1]

    # North potential spike at q3
    #
    # q3    q2    q1
    #  |     |     |
    # t1 -- s1 -- r1
    #        |
    #       t2
    row = N
    for i, col in enumerate(range(N, 2, -1)):
        q1 = 2 * N + i
        q2 = q1 + 1
        q3 = q2 + 1
        r1 = interior_index(N, row, col)
        s1 = interior_index(N, row, col - 1)
        t1 = interior_index(N, row, col - 2)
        t2 = interior_index(N, row - 1, col - 1)

        # Calculate c[q3,t2,s1] using Kirchoff's Law
        # Already have u[q3,s1]
        u[q3, r1] = -c[q3, q1, r1] / K[q1, r1]
        c[q3, s1, r1] = K[s1, r1] * (u[q3, s1] - u[q3, r1])  # Ohm's Law
        c[q3, t1, s1] = -c[q3, q3, t1]
        c[q3, t2, s1] = -c[q3, q2, s1] + c[q3, t1, s1] + c[q3, s1, r1]
        K[t2, s1] = -c[q3, t2, s1] / u[q3, s1]

    # West potential spike at q3
    #
    # q3 -- t1
    #        |
    # q2 -- s1 -- t2
    #        |
    # q1 -- r1
    col = 1
    for i, row in enumerate(range(1, N - 1)):
        q1 = 4 * N - i - 1
        q2 = q1 - 1
        q3 = q2 - 1
        r1 = interior_index(N, row, col)
        s1 = interior_index(N, row + 1, col)
        t1 = interior_index(N, row + 2, col)
        t2 = interior_index(N, row + 1, col + 1)

        # Calculate c[q3,t2,s1] using Kirchoff's Law
        # Already have u[q3,s1]
        u[q3, r1] = -c[q3, q1, r1] / K[q1, r1]
        c[q3, s1, r1] = K[s1, r1] * (u[q3, s1] - u[q3, r1])  # Ohm's Law
        c[q3, t1, s1] = -c[q3, q3, t1]
        c[q3, t2, s1] = -c[q3, q2, s1] + c[q3, t1, s1] + c[q3, s1, r1]
        K[t2, s1] = -c[q3, t2, s1] / u[q3, s1]

    K = K + K.T  # Make Symmetric
    K = K - np.diag(np.sum(K, axis=1))  # Fill in diagonal entries
    return K


def rotation_vector(N):
    """
    A N-vector giving the indexes resulting from rotating the electrical graph by
    45 degrees counterclockwise. E.g. for use with Kirchhoff matrices.
    """
    # "top" and "bot(tom)" refer to the "matrix notation" for the permutation group:
    #   1 2 3 4
    #   2 3 4 1
    # means
    #   1 -> 2, 2 -> 3, 3 -> 4, 4 -> 1.
    # See "Aluffi: Algebra Chapter 0" or any other Algebra textbook.
    # First, set the boundary nodes.
    top = np.arange(num_nodes(N))
    bot = np.zeros_like(top)
    bot[boundary_slice_S(N)] = top[boundary_slice_W(N)]
    bot[boundary_slice_E(N)] = top[boundary_slice_S(N)]
    bot[boundary_slice_N(N)] = top[boundary_slice_E(N)]
    bot[boundary_slice_W(N)] = top[boundary_slice_N(N)]
    # Now the interior nodes.
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            bot[interior_index(N, i, j)] = top[interior_index(N, (N + 1) - j, i)]
    return bot


def boundary_rotation_vector(N):
    """
    See the docstring for `rotation_vector`.
    For sue with, e.g. response maps $\Lambda$.
    """
    top = np.arange(4 * N)
    bot = np.zeros_like(top)
    bot[boundary_slice_S(N)] = top[boundary_slice_W(N)]
    bot[boundary_slice_E(N)] = top[boundary_slice_S(N)]
    bot[boundary_slice_N(N)] = top[boundary_slice_E(N)]
    bot[boundary_slice_W(N)] = top[boundary_slice_N(N)]
    return bot


def permutation_matrix(vec):
    """
    Build a permutation matrix $P$ from a permutation vector $v$.
    $$
        Py = v
    $$
    where $y = (0, 1, 2, .., N)$
    """
    M = np.zeros([vec.size, vec.size])
    for j in range(vec.size):
        M[j, vec[j]] = 1
    return M


NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


class Current:
    # Index by southern node for vertical edges, western node for horizontal edges
    def __init__(self, N):
        self.N = N
        self.vec = np.zeros([num_nodes(N), 2])

    def get(self, node, direction):
        if direction == NORTH or direction == EAST:
            return self.vec[node, direction]

        row, col = node_to_row_col(self.N, node)
        if direction == SOUTH:
            return self.get(interior_index(self.N, row - 1, col), NORTH)
        if direction == WEST:
            return self.get(interior_index(self.N, row, col - 1), EAST)

    def set(self, node, direction, val):
        if direction == NORTH or direction == EAST:
            self.vec[node, direction] = val

        row, col = node_to_row_col(self.N, node)
        if direction == SOUTH:
            self.set(interior_index(self.N, row - 1, col), NORTH, val)
        if direction == WEST:
            self.set(interior_index(self.N, row, col - 1), EAST, val)


def boundary_response_to_full_current(N, response):
    """
    Canvert a size [4*N] vector of boundary currents to a size [num_nodes][4] matrix
    meant to hold currents of all edges. First index of c is start node, second index
    should be one of SOUTH, EAST, NORTH, WEST
    """
    c = Current(N)
    for node in range(N):  # South
        c.set(node, NORTH, response[node])
    for node in range(N, 2 * N):  # East
        c.set(node, WEST, response[node])
    for node in range(2 * N, 3 * N):  # North
        c.set(node, SOUTH, response[node])
    for node in range(3 * N, 4 * N):  # North
        c.set(node, EAST, response[node])
    return c


def curtis_morrow(N, Lambda):
    K = np.zeros([K_dim(N), K_dim(N)])

    K_rot_mat = permutation_matrix(rotation_vector(N))
    Lambda_rot_mat = permutation_matrix(boundary_rotation_vector(N))

    if N > 0:
        for rotations in range(4):
            for lvl in range(1):

                y_N = np.zeros(N)
                y_N[lvl] = 1

                psi_W = Lambda[boundary_slice_W(N), boundary_slice_N(N)] @ y_N
                x_E = np.linalg.solve(
                    Lambda[boundary_slice_W(N), boundary_slice_E(N)], -psi_W
                )
                x = np.zeros(4 * N)
                x[boundary_slice_N(N)] = y_N
                x[boundary_slice_E(N)] = x_E

                current = boundary_response_to_full_current(N, Lambda @ x)
                # Current = 0 on West.
                # By harmonic continuation, current = 0 on all interior nodes up
                # to triangle starting at q_lvl

                # Calculate conductances in zig-zag pattern
                for edge in range(lvl + 1):
                    K = update(N, K, current, lvl, edge)

            # Rotate
            K = K_rot_mat @ K @ K_rot_mat.T
            Lambda = Lambda_rot_mat @ Lambda @ Lambda_rot_mat.T

    K = K - np.diag(np.sum(K, axis=1))  # Fill in diagonal entries
    return K


def update(N, K, current, lvl, edge):
    # Using Curtis and Morrow's notation
    q1 = 2 * N
    r = interior_index(N, N, N)

    K[q1, r] = -current.get(r, NORTH)
    K[r, q1] = K[q1, r]

    return K

import numpy as np
import eit

rng = np.random.default_rng(0)


def test_rotation_vector():
    np.testing.assert_array_equal(
        eit.rotation_vector(4),
        np.array(
            [12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, # Boundary
             28, 24, 20, 16, 29, 25, 21, 17, 30, 26, 22, 18, 31, 27, 23, 19] # Interior
        )
    )  # fmt: skip


def test_permutation_matrix():
    y = np.arange(32)
    v = eit.rotation_vector(4)
    M = eit.permutation_matrix(v)
    np.testing.assert_allclose(M @ y, v)


def test_index():
    assert eit.interior_index(4, 0, 1) == 0
    assert eit.interior_index(4, 1, 5) == 4
    assert eit.interior_index(4, 5, 2) == 10
    assert eit.interior_index(4, 1, 0) == 15
    assert eit.interior_index(4, 2, 2) == 21


def test_neighbors():
    assert eit.neighbors(4, 0) == [16]
    assert eit.neighbors(4, 1) == [17]
    assert eit.neighbors(4, 4) == [19]
    assert eit.neighbors(4, 8) == [31]
    assert eit.neighbors(4, 12) == [28]
    assert eit.neighbors(4, 28) == [24, 12, 29, 11]
    assert eit.neighbors(4, 21) == [17, 20, 22, 25]


def test_rectangular_inversion():
    def rectangular_inversion(N):
        adj = eit.build_adjacency_matrix(N)
        K = eit.build_kirchoff_matrix_from_adjacency_mask(rng.random(adj.shape), adj)
        Lambda = eit.forward(N, K)
        recovered_K = eit.curtis_morrow(N, Lambda)
        np.testing.assert_allclose(K, recovered_K)

    rectangular_inversion(0)
    rectangular_inversion(1)


#     # rectangular_inversion(2)
#     # rectangular_inversion(3)
#     # rectangular_inversion(4)


def test_interior_index():
    assert eit.interior_index(4, 1, 1) == 16
    assert eit.interior_index(4, 2, 2) == 21
    assert eit.interior_index(4, 4, 3) == 30

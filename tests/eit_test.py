import numpy as np
import eit

rng = np.random.default_rng(0)


def rectangular_inversion(N):
    adj = eit.build_adjacency_matrix(N)
    K = eit.build_kirchoff_matrix_from_adjacency_mask(rng.random(adj.shape), adj)
    Lambda = eit.forward(N, K)
    recovered_K = eit.curtis_morrow(N, Lambda)
    np.testing.assert_allclose(K, recovered_K)


def test_rectangular_inversion():
    rectangular_inversion(0)
    rectangular_inversion(1)
    rectangular_inversion(2)
    # rectangular_inversion(3)
    # rectangular_inversion(4)


def test_interior_index():
    assert eit.interior_index(4, 1, 1) == 16
    assert eit.interior_index(4, 2, 2) == 21
    assert eit.interior_index(4, 4, 3) == 30

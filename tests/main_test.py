from main import interior_index


def test_interior_index():
    assert interior_index(4, 1, 1) == 16
    assert interior_index(4, 2, 2) == 21
    assert interior_index(4, 4, 3) == 30

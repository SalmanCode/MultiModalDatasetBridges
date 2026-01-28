import pytest
from utils import compute_girder_spacing

@pytest.mark.parametrize(
    "width, min_spacing, expected_num, expected_spacing",
    [
        (8.0, 3.5, 2, 4.0),
        (10.5, 3.5, 3, 3.5),
        (14.0, 3.5, 4, 3.5),
        (12.0, 4.0, 3, 4.0),
    ],
)
def test_compute_girder_spacing(width, min_spacing, expected_num, expected_spacing):
    num, spacing = compute_girder_spacing(width, min_spacing)
    assert num == expected_num
    assert spacing == pytest.approx(expected_spacing, rel=1e-6)
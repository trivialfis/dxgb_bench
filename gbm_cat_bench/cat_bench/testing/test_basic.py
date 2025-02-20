import pytest

from cat_bench import AVAILABLE_ALGOS, make_estimator


@pytest.mark.parametrize("algo", AVAILABLE_ALGOS)
def test_ames_housing(algo: str) -> None:
    est = make_estimator(algo, 4, depth=6)
    result = est.fit("ames_housing")
    assert result is not None

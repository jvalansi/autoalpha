import importlib
import pytest

PACKAGES = [
    "core",
    "data",
    "labeling",
    "backtest",
    "evaluation",
    "research",
    "execution",
    "strategies",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_importable(package):
    importlib.import_module(package)

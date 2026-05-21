import importlib
import pytest

PACKAGES = [
    "autoalpha.core",
    "autoalpha.data",
    "autoalpha.labeling",
    "autoalpha.backtest",
    "autoalpha.evaluation",
    "autoalpha.research",
    "autoalpha.execution",
    "autoalpha.strategies",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_importable(package):
    importlib.import_module(package)

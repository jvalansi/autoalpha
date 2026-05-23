"""Tests for code_validator — AST safety checks and predict() wrapping."""
from __future__ import annotations

import pytest

from autoalpha.research.code_validator import (
    CodeValidationError,
    validate_predict_body,
    wrap_predict_body,
)

_SAFE_BODY = """\
        if bar_data.empty or bar_date is None:
            return {}
        scores = bar_data["ret_21d"].dropna()
        if scores.empty:
            return {}
        top = scores.nlargest(max(1, len(scores) // 5)).index.tolist()
        w = 1.0 / len(top)
        return {t: w for t in top}
"""


class TestValidatePredictBody:
    def test_safe_body_passes(self):
        validate_predict_body(_SAFE_BODY)  # should not raise

    def test_import_rejected(self):
        body = "        import os\n        return {}"
        with pytest.raises(CodeValidationError, match="Import"):
            validate_predict_body(body)

    def test_from_import_rejected(self):
        body = "        from pathlib import Path\n        return {}"
        with pytest.raises(CodeValidationError, match="ImportFrom"):
            validate_predict_body(body)

    def test_exec_name_rejected(self):
        body = "        exec('print(1)')\n        return {}"
        with pytest.raises(CodeValidationError, match="exec"):
            validate_predict_body(body)

    def test_eval_name_rejected(self):
        body = "        x = eval('1+1')\n        return {}"
        with pytest.raises(CodeValidationError, match="eval"):
            validate_predict_body(body)

    def test_open_name_rejected(self):
        body = "        f = open('/etc/passwd')\n        return {}"
        with pytest.raises(CodeValidationError, match="open"):
            validate_predict_body(body)

    def test_os_attribute_rejected(self):
        body = "        import os\n        os.system('rm -rf /')\n        return {}"
        with pytest.raises(CodeValidationError):
            validate_predict_body(body)

    def test_syntax_error_rejected(self):
        body = "        def bad(:\n            return {}"
        with pytest.raises(CodeValidationError, match="Syntax"):
            validate_predict_body(body)

    def test_body_too_long_rejected(self):
        body = "        x = 1\n" * 300  # well over 4000 chars
        with pytest.raises(CodeValidationError, match="too long"):
            validate_predict_body(body)

    def test_global_statement_rejected(self):
        body = "        global foo\n        return {}"
        with pytest.raises(CodeValidationError, match="Global"):
            validate_predict_body(body)

    def test_nonlocal_statement_rejected(self):
        body = "        nonlocal x\n        return {}"
        with pytest.raises(CodeValidationError, match="Nonlocal"):
            validate_predict_body(body)


class TestWrapPredictBody:
    def test_wrap_produces_valid_python(self):
        source = wrap_predict_body(_SAFE_BODY)
        compile(source, "<test>", "exec")  # must not raise

    def test_wrap_contains_class(self):
        source = wrap_predict_body(_SAFE_BODY, class_name="MyStrat")
        assert "class MyStrat(Strategy):" in source

    def test_wrap_contains_strategy_instance(self):
        source = wrap_predict_body(_SAFE_BODY)
        assert "strategy = " in source

    def test_wrap_imports_pandas(self):
        source = wrap_predict_body(_SAFE_BODY)
        assert "import pandas as pd" in source

    def test_roundtrip_execute(self):
        """Wrapped source should be importable and strategy.predict() callable."""
        import importlib.util
        import sys
        import types

        import pandas as pd

        source = wrap_predict_body(_SAFE_BODY)
        mod = types.ModuleType("_test_generated")
        exec(compile(source, "<test>", "exec"), mod.__dict__)  # noqa: S102
        strategy = mod.strategy
        bar = pd.DataFrame({"ret_21d": [0.05, 0.02, -0.01]}, index=["A", "B", "C"])
        result = strategy.predict(bar, bar_date=pd.Timestamp("2024-01-15"))
        assert isinstance(result, dict)
        assert all(v > 0 for v in result.values())
        assert abs(sum(result.values()) - 1.0) < 1e-9

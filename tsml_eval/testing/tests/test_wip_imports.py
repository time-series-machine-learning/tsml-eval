"""Tests that modules outside of _wip do not import from _wip."""

import ast
from importlib.util import find_spec

from tsml_eval.testing.tests import ALL_TSML_EVAL_MODULES

ALLOWED_WIP_IMPORTS = [
    "tsml_eval._wip.pf._pf2",
    "tsml_eval._wip.rt.transformations.collection.imbalance._tsmote",
    "tsml_eval._wip.unequal_length._arsenal4",
    "tsml_eval._wip.unequal_length._drcif6",
    "tsml_eval._wip.unequal_length._fresh_prince",
    "tsml_eval._wip.unequal_length._rocket",
    "tsml_eval._wip.unequal_length._stc7",
    "tsml_eval._wip.unequal_length._tde7",
]


def _is_wip(name):
    return name == "tsml_eval._wip" or name.startswith("tsml_eval._wip.")


def _wip_imports(module):
    with open(find_spec(module).origin, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if _is_wip(node.module):
                names = [node.module]
            else:
                names = [f"{node.module}.{alias.name}" for alias in node.names]
        else:
            continue
        imports.update(name for name in names if _is_wip(name))
    return imports


def test_no_wip_imports():
    """Test that modules outside of _wip only make the allowed imports from _wip.

    _wip is excluded from releases, so importing it from any other module fails when
    tsml-eval is installed. Remove an allowed import once it is no longer used.
    """
    wip_imports = {}
    for module in ALL_TSML_EVAL_MODULES:
        for name in _wip_imports(module):
            wip_imports.setdefault(name, []).append(module)

    not_allowed = {k: v for k, v in wip_imports.items() if k not in ALLOWED_WIP_IMPORTS}
    assert not not_allowed, f"Modules outside of _wip import from _wip: {not_allowed}"

    unused = [name for name in ALLOWED_WIP_IMPORTS if name not in wip_imports]
    assert not unused, f"Allowed _wip imports are no longer used: {unused}"

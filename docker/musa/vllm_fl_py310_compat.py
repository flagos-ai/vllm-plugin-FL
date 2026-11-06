"""Py3.10 compatibility shim for vLLM 0.32 on the MUSA stack.

vLLM 0.32 declares ``python_requires >= 3.11`` while ``torch_musa`` only ships
cp310 wheels, so the MUSA image installs the empty wheel with
``--ignore-requires-python`` and back-fills the 3.11-only names that vLLM and
its dependencies import at module load time:

* ``typing.Never`` / ``Self`` / ``assert_never`` / ``NotRequired`` /
  ``Required`` / ``LiteralString`` -- from typing_extensions.
* ``datetime.UTC`` -- alias of ``timezone.utc``.
* ``enum.StrEnum`` (mistral_common) and ``enum.global_enum``
  (python-multipart >= 0.0.21) -- minimal local implementations.

The module is imported from a ``.pth`` file so it runs before any vLLM import.

Do NOT back-fill ``TypeAliasType``/``Protocol``/``ParamSpec``/``TypeVarTuple``:
typing_extensions treats ``hasattr(typing, "TypeAliasType")`` as its "PEP 695
is implemented" probe and then delegates to the stdlib implementation, which on
3.10 rejects the new keyword arguments (e.g. ``TypeVar(infer_variance=...)``)
and breaks torch import.
"""

try:
    import typing as _t

    import typing_extensions as _te

    for _name in (
        "Never",
        "Self",
        "assert_never",
        "NotRequired",
        "Required",
        "LiteralString",
    ):
        if not hasattr(_t, _name) and hasattr(_te, _name):
            setattr(_t, _name, getattr(_te, _name))
except Exception:
    pass

try:
    import datetime as _dt

    if not hasattr(_dt, "UTC"):
        _dt.UTC = _dt.timezone.utc
except Exception:
    pass

try:
    import enum as _enum

    class _StrEnum(str, _enum.Enum):
        def __str__(self):
            return str(self.value)

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

    if not hasattr(_enum, "StrEnum"):
        _enum.StrEnum = _StrEnum

    if not hasattr(_enum, "global_enum"):

        def _global_enum(cls):
            return cls

        _enum.global_enum = _global_enum
except Exception:
    pass

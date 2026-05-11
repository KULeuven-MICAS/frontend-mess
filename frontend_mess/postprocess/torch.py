from io import StringIO
from itertools import count
import re
import subprocess
from typing import Sequence


_DENSE_RESOURCE_RE = re.compile(
    r'dense_resource<[^>]*>\s*:\s*tensor<([^>]+)>'
)

_DENSE_RE = re.compile(
    r'dense<([^>]*)>\s*:\s*tensor<([^>]+)>'
)


def _elt_type(tensor_inner: str) -> str:
    return tensor_inner.rsplit('x', 1)[-1].strip()


_INT_TYPE_RE = re.compile(r'([ius]?)i(\d+)')


def _splat_value(counter_value: int, elt: str) -> str:
    if elt == 'i1':
        return '1'
    m = _INT_TYPE_RE.fullmatch(elt)
    if m:
        max_val = 2 ** (int(m.group(2)) - (0 if m.group(1) == 'u' else 1)) - 1
        return str(((counter_value - 1) % max_val) + 1)
    if elt == 'index':
        return str(counter_value)
    return f"{float(counter_value)}"


def _make_counter():
    c = count(1)
    return lambda: next(c)


def nullify_dense_resources(input_module: str) -> str:
    """Replace each `dense_resource<...>` with a unique splat to keep the IR
    compact without making distinct weights look identical to CSE."""
    nxt = _make_counter()
    def repl(m: re.Match) -> str:
        tensor_inner = m.group(1)
        return f"dense<{_splat_value(nxt(), _elt_type(tensor_inner))}> : tensor<{tensor_inner}>"
    sub = _DENSE_RESOURCE_RE.sub(repl, input_module)
    return run_mlir_opt(StringIO(sub), ["--linalg-generalize-named-ops"]).getvalue()


def nullify_dense_constants(input_module: str) -> str:
    """Replace each non-splat inline `dense<...>` (hex blob or array literal)
    with a unique splat. Leaves already-splat dense literals alone."""
    nxt = _make_counter()
    def repl(m: re.Match) -> str:
        body = m.group(1)
        tensor_inner = m.group(2)
        if '[' not in body and '"' not in body and ',' not in body:
            return m.group(0)
        return f"dense<{_splat_value(nxt(), _elt_type(tensor_inner))}> : tensor<{tensor_inner}>"
    return _DENSE_RE.sub(repl, input_module)


def run_mlir_opt(stream: StringIO, opts: Sequence[str])-> StringIO:
    input_bytes = stream.getvalue().encode()

    process = subprocess.Popen(
        ["mlir-opt", *opts],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = process.communicate(input=input_bytes)

    if process.returncode != 0:
        raise RuntimeError(f"mlir-opt failed: {stderr.decode()}")

    return StringIO(stdout.decode())

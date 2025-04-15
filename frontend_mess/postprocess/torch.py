from io import StringIO
import re
import subprocess
from typing import Sequence

def nullify_dense_resources(input_module: str) -> str:
    """
    Removes all dense_resources and naively changes them to 0
    """
    regex_float = r"dense_resource<\S*float\S*>"
    subst_float = "dense<0.0>"
    regex_int = r"dense_resource<\S*int\S*>"
    subst_int = "dense<0>"

    sub_module = re.sub(regex_float, subst_float, input_module, 0, re.MULTILINE)
    sub_module = re.sub(regex_int, subst_int, sub_module, 0, re.MULTILINE)
    module_stream = StringIO(sub_module)
    module_stream = run_mlir_opt(module_stream, ["--test-linalg-transform-patterns=test-generalize-pad-tensor","--linalg-generalize-named-ops", "-test-linalg-elementwise-fusion-patterns=fuse-multiuse-producer","--mlir-print-op-generic"])
    return module_stream.getvalue()


def run_mlir_opt(stream: StringIO, opts: Sequence[str])-> StringIO:
    input_bytes = stream.getvalue().encode()

    process = subprocess.Popen(
        ["mlir-opt", *opts],  # Command
        stdin=subprocess.PIPE,  # Pass input via stdin
        stdout=subprocess.PIPE,  # Capture output
        stderr=subprocess.PIPE  # Capture errors (optional)
    )

    # Communicate input and get output
    stdout, stderr = process.communicate(input=input_bytes)

    # Check for errors
    if process.returncode != 0:
        raise RuntimeError(f"mlir-opt failed: {stderr.decode()}")

    # Convert output bytes to StringIO
    return StringIO(stdout.decode())

import subprocess
import tempfile
from pathlib import Path

from frontend_mess.postprocess.torch import nullify_dense_resources


TOSA_TO_LINALG_PASSES = [
    "--pass-pipeline=builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith{include-apply-rescale=true},tosa-to-tensor),canonicalize)",
]


def tflite_to_tosa_bytecode(tflite_path: Path, out_path: Path) -> None:
    subprocess.run(
        ["iree-import-tflite", str(tflite_path), "-o", str(out_path)],
        check=True,
        capture_output=True,
    )


def mlir_opt(input_path: Path, opts: list[str]) -> str:
    result = subprocess.run(
        ["mlir-opt", str(input_path), *opts],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode()


def tflite_to_linalg(tflite_path: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        bytecode = Path(tmp) / "model.mlirbc"
        tflite_to_tosa_bytecode(tflite_path, bytecode)
        linalg_mlir = mlir_opt(bytecode, TOSA_TO_LINALG_PASSES)
    return nullify_dense_resources(linalg_mlir)

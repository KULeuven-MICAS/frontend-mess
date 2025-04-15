# Machine-Learning Front-ends are a Mess

For people working on the deployment of machine-learning algorithms, importing machine-learning algorithms or neural nets from established frameworks or serialization formats, can be quite daunting.
I created this repo because I was frustrated with the difficulty of getting neural nets into the MLIR compilation framework, which should not be considered the core business of MLIR, though it can be considered as one of its excellent use cases.

Why is importing a neural net in MLIR from these frameworks so frustrating?:

* Dependency hell: the mere installation process and getting the right packages in itself can be quite difficult. 
Many of the machine-learning packages in use have stringent and bleeding-edge dependencies on MLIR (looking at you TensorFlow) or python (Neither TensorFlow, nor PyTorch support python >= 3.11), or some slow package manager (looking at you Conda). It is impossible to install new installations of PyTorch alongside Tensorflow in the same environment, because PyTorch has a dependency on numpy > 2.0, which TensorFlow does not support. For a long time, IREE had no stable versioning scheme, and torch-mlir still doesn't seem to have one.
* So many dialects: For each machine-learning framework that gets created, a new set of dialects gets created that seems to be doing more or less the same thing.
To give you a clue, we now have ONNX, TOSA, StableHLO, Torch, Torch_c, which all seem to be doing the same?
* New frameworks for machine-learning seem to keep popping up, and all of them have different names for their layers and different semantics.

How does this repo help?

* This repo uses `pixi` as a faster alternative to conda, which explicitly locks versions of all tools used. If it works in CI, it should also work on your machine, with a single installation command.
* I'm documenting my findings here as I go.

## Usage of this repo

### Installation

[Get pixi](https://pixi.sh/latest/)

```sh
git clone git@github.com:KULeuven-MICAS/frontend-mess.git
cd frontend-mess
pixi shell
```

This will download all packages required to run the tests/examples in this repo.
pixi shell needs to be invoked once in each new terminal session you open.

### Examples

Currently, the examples in this repo only work with PyTorch models.
It uses vanilla torch-mlir to go to linalg on tensors.
This environment can convert manually made torch modules, imported torchvision networks (also just modules), and huggingface transformers networks (also just modules).

From within the pixi shell:
```sh
python examples/nnmodule.py
```

### Tests

This repo only ships with Lit tests, so from within the pixi shell
```sh
lit -vv tests/filecheck
```

## Things I learned along the way:

### PyTorch

It Took me a while to figure out the difference between all of these tools; but I think this is the gist:

[`torch.fx`](https://arxiv.org/pdf/2112.08429) is a simplified successor to [`torchscript`](https://pytorch.org/docs/1.9.0/jit.html) that performs graph capture for torch graphs. 
torchscript is/was? an initial aim at providing graph capture in torch, but it became too complex, so people gave up on it. 
Also people had to change their networks to be traceable, which they didn't like doing. [video-ref](https://www.youtube.com/watch?v=5FNHwPIyHr8) ("Pytorch has always been about dynamism and eager mode etc." [video-ref](https://www.youtube.com/watch?v=JUwAuZ9Y8ek))

`torch 2.x` now (by default) is able to do partial graph captures and send this off for compilation, instead of forcing users to write their NNs differently. 
It breaks for example at IO operations or conditional control flow etc. 
This system is called `torchdynamo`. 
Each identified graph region can be traced by torch.fx to be sent of to a compilation backend. 
Compiled code gets injected back into python through a system defined in [PEP 523](https://peps.python.org/pep-0523/). 
Recurring code snippets are cached (through a system of guards) for improving JIT runtime performance.
The default to do this is to use `torch.compile`. `torch.export` can also be used, and it forces `torchdynamo` to trace the entire graph without breaks (which is useful if you just want to export the entire graph, and which is required if you want to run the network on a system that does not have python - i.e. AOT graph capture).
Most (all?) of `torchdynamo` is completely available in python.

[`torch-inductor`](https://github.com/pytorch/pytorch/tree/main/torch/_inductor) is a gpu backend for `dynamo` that uses torch fx for graph capture, converts it to triton DSL to compile to CUDA kernels to nvcc to compile to NVIDIA gpus (and HIP for ROCM?) to run accelerated kernels on GPU.

[iree-turbine](https://github.com/iree-org/iree-turbine) can act as an MLIR-native replacement for `torch-inductor` (for use with `torch.compile` or can act as an AOT export backend, similar to `torch.export`.

[torch-mlir](https://github.com/llvm/torch-mlir) is an LLVM incubator project that can perform `torch.fx` graph capture, and convert these graphs into `torch` (of which `torch.aten`) and `torch_c` dialects.
`torch-mlir`'s tests run on a reference backend (i.e. converts to a set of just upstream dialects to convert to LLVM and run through `mlir-cpu-runner` pretty much)
`iree-turbine` uses a version of this, that uses additional IREE low-level dialects to facilitate loading, exporting and handling large weights inside these models

### HuggingFace Transformers

[huggingface transformers](https://github.com/huggingface/transformers) is a library of python functions that call pytorch, gguf, tensorflow, jax, ... models for you with a few simple lines of code. 
It contains helper functions for tokenizing python strings etc, (to convert a string into a tensor, which you can feed into your pytorch, gguf, tensorflow,... model)
Or for pushing a query through an LLM. 
It also allows to download models from the [huggingface hub](https://huggingface.co/models) (a glorified github that uses LFS to store enormous amounts of weights files together with the model architecture.
This is very similar to some of the networks in torchvision (essentially, these are preset networks, which you can download trained versions from through a very simple python script (1 line of python, literally).
huggingface transformers allows to get a pytorch model (if the model from the hub you are using is implemented in pytorch), which you can then trace through `iree-turbine`, or `torch.fx`.


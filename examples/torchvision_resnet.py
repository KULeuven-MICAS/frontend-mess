import torch
from torchvision import models
from torch_mlir import fx
from frontend_mess.postprocess.torch import nullify_dense_resources


if __name__ == "__main__":
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    example_args = torch.ones(1, 3, 224, 224)

    module = fx.export_and_import(
        resnet18,
        example_args,
        output_type="linalg-on-tensors",
        func_name=resnet18.__class__.__name__,
    )
    print(nullify_dense_resources(str(module)))

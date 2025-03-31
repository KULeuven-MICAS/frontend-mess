import torch
import torchvision.models as models
from torch_mlir import fx


if __name__ == "__main__":

    resnet18 = models.resnet18(pretrained=True).eval()
    example_args = torch.ones(1, 3, 224, 224)

    module = fx.export_and_import(
        resnet18,
        example_args,
        output_type="linalg-on-tensors",
        func_name=resnet18.__class__.__name__,
    )

import torch
import torchvision.models as models
from torch_mlir import fx

resnet18 = models.resnet18(pretrained=True).eval()
module = fx.export_and_import(
    resnet18,
    torch.ones(1, 3, 224, 224),
    output_type="linalg-on-tensors",
    func_name=resnet18.__class__.__name__,
)
with open("import.mlir", "w") as f:
    f.write(str(module))

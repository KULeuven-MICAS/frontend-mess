import torch
import torchvision.models as models
from torch_mlir import fx

def get_linear_layer():    
    class LinearModule(torch.nn.Module):
      def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
    
      def forward(self, input):
        return (input @ self.weight) + self.bias
    
    linear_module = LinearModule(4, 3)
    return linear_module, torch.randn(4)

#resnet18 = models.resnet18(pretrained=True).eval()
#module = fx.export_and_import(
#    resnet18,
#    torch.ones(1, 3, 224, 224),
#    output_type="linalg-on-tensors",
#    func_name=resnet18.__class__.__name__,
#)

module = str(fx.export_and_import(
    get_linear_layer()[0],
    get_linear_layer()[1],
    output_type="linalg-on-tensors",
    func_name="linear",
))


import re

regex_float = r"dense_resource<\S*float\S*>"
subst_float = "dense<0.0>"
regex_int = r"dense_resource<\S*int\S*>"
subst_int = "dense<0>"

# You can manually specify the number of replacements by changing the 4th argument
result = re.sub(regex_float, subst_float, module, 0, re.MULTILINE)
result = re.sub(regex_int, subst_int, result , 0, re.MULTILINE)

#with open("import.mlir", "w") as f:
#    f.write(str(result))
print(result)




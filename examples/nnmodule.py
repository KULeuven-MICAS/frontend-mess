import torch
from torch_mlir import fx
from frontend_mess.postprocess.torch import nullify_dense_resources

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

if __name__ == "__main__":
    module = str(fx.export_and_import(
        *get_linear_layer(),
        output_type="linalg-on-tensors",
        func_name="linear_layer",
    ))
    module = nullify_dense_resources(module)
    print(module)

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_mlir import fx
from frontend_mess.postprocess.torch import nullify_dense_resources
import torch

modelname = "openai/whisper-small"
tokenizer = AutoTokenizer.from_pretrained(modelname)

# Some of the options here affect how the model is exported. See the test cases
# at https://github.com/nod-ai/SHARK-TestSuite/tree/main/e2eshark/pytorch/models
# for other options that may be useful to set.
model = AutoModelForCausalLM.from_pretrained(
    modelname,
    output_attentions=False,
    output_hidden_states=False,
    attn_implementation="eager",
    torchscript=False,
)

# This is just a simple demo to get some data flowing through the model.
# Depending on this model and what input it expects (text, image, audio, etc.)
# this might instead use a specific Processor class. For Whisper,
# WhisperProcessor runs audio input pre-processing and output post-processing.
example_prompt = "Hello world!"
example_encoding = tokenizer(example_prompt, return_tensors="pt")
example_input = example_encoding["input_ids"].cpu()
example_args = (example_input,)

module = fx.export_and_import(
    model,
    *example_args, # Pass tensors directly, not as a dictionary
    output_type="linalg-on-tensors",
    func_name=model.__class__.__name__,
    import_symbolic_shape_expressions = True
)
print(nullify_dense_resources(str(module)))

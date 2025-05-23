from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from frontend_mess.postprocess.torch import nullify_dense_resources
from torch_mlir import fx

model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
prompt = "Once upon a time there was"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
# Generate completion
module = fx.export_and_import(
    model,
    input_ids, # Pass tensors directly, not as a dictionary
    output_type="linalg-on-tensors",
    func_name=model.__class__.__name__,
    import_symbolic_shape_expressions = True
)
print(nullify_dense_resources(str(module)))
exit(0)
output = model.generate(input_ids, max_length = 1000, num_beams=1)
# Decode the completion
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
# Print the generated text
print(output_text)

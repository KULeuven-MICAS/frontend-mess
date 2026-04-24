from transformers import AutoTokenizer, AutoModelForMaskedLM
from frontend_mess.postprocess.torch import nullify_dense_resources
from torch_mlir import fx

if __name__ == "__main__":
    # Load tokenizer and model
    model = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model)
    # `attn_implementation="eager"` forces the manual matmul-softmax-matmul
    # decomposition. The SDPA path lowers to `tm_tensor.attention`, which the
    # bundled mlir-opt does not register.
    model = AutoModelForMaskedLM.from_pretrained(model, attn_implementation="eager")
    model.eval()

    # Create proper input for BERT model
    # BERT expects input_ids, attention_mask, and token_type_ids
    text = "This is an [MASK] sentence."
    encoded_input = tokenizer(text, return_tensors="pt")
    # Export the model with individual tensors as arguments
    module = fx.export_and_import(
        model,
        **encoded_input,
        output_type="linalg-on-tensors",
        func_name=model.__class__.__name__,
    )
    print(nullify_dense_resources(str(module))) 

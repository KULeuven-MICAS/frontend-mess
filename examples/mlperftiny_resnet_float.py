from frontend_mess.utils.fetch import fetch_model
from frontend_mess.postprocess.tflite import tflite_to_linalg


RESNET_FLOAT_URL = (
    "https://github.com/mlcommons/tiny/blob/master/benchmark/training/"
    "image_classification/trained_models/pretrainedResnet_large_float.tflite"
)


if __name__ == "__main__":
    model_path = fetch_model(RESNET_FLOAT_URL)
    print(tflite_to_linalg(model_path))

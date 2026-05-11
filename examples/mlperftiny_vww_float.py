from frontend_mess.utils.fetch import fetch_model
from frontend_mess.postprocess.tflite import tflite_to_linalg


VWW_FLOAT_URL = (
    "https://github.com/mlcommons/tiny/blob/master/benchmark/training/"
    "visual_wake_words/trained_models/vww_96_float.tflite"
)


if __name__ == "__main__":
    model_path = fetch_model(VWW_FLOAT_URL)
    print(tflite_to_linalg(model_path))

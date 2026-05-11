from frontend_mess.utils.fetch import fetch_model
from frontend_mess.postprocess.tflite import tflite_to_linalg


AD01_FLOAT_URL = (
    "https://github.com/mlcommons/tiny/blob/master/benchmark/training/"
    "anomaly_detection/trained_models/ad01_fp32.tflite"
)


if __name__ == "__main__":
    model_path = fetch_model(AD01_FLOAT_URL)
    print(tflite_to_linalg(model_path))

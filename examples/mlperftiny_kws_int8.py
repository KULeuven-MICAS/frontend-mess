from frontend_mess.utils.fetch import fetch_model
from frontend_mess.postprocess.tflite import tflite_to_linalg


KWS_INT8_URL = (
    "https://github.com/mlcommons/tiny/blob/master/benchmark/training/"
    "keyword_spotting/trained_models/kws_ref_model.tflite"
)


if __name__ == "__main__":
    model_path = fetch_model(KWS_INT8_URL)
    print(tflite_to_linalg(model_path))

from importlib import import_module

import numpy as np


EMBEDDING_DIMENSION = 512
MODEL_NAME = "buffalo_l"


def load_arcface_model(allowed_modules=None, threads=None):
    try:
        insightface_app = import_module("insightface.app")
        onnxruntime = import_module("onnxruntime")
    except ImportError as error:
        raise RuntimeError("ArcFace extraction requires InsightFace and ONNX Runtime. Install them with: pip install insightface onnxruntime") from error

    # on CPUs with low-power cores, a few threads is much faster than letting ONNX Runtime use all of them
    session_options = onnxruntime.SessionOptions()
    if threads is not None:
        session_options.intra_op_num_threads = threads
    # use the GPU through DirectML when onnxruntime-directml is installed, otherwise stay on the CPU
    use_gpu = "DmlExecutionProvider" in onnxruntime.get_available_providers()
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    face_analysis = getattr(insightface_app, "FaceAnalysis")
    model = face_analysis(name=MODEL_NAME, providers=providers, allowed_modules=allowed_modules, sess_options=session_options)
    model.prepare(ctx_id=0 if use_gpu else -1)
    return model


def extract_arcface_embedding(model, image):
    rgb_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr_image = rgb_image[:, :, ::-1].copy()
    faces = model.get(bgr_image)

    if len(faces) != 1:
        raise ValueError(f"Expected exactly one face in the image, but detected {len(faces)}.")

    embedding = np.asarray(faces[0].embedding, dtype=np.float32).reshape(-1)
    if embedding.size != EMBEDDING_DIMENSION:
        raise ValueError(f"Expected a {EMBEDDING_DIMENSION}-D ArcFace embedding, received {embedding.size} values.")

    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("ArcFace returned an invalid or zero-length embedding.")

    return embedding / norm
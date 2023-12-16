"""
Mapping of the shared libraries with custom ONNX operations
library name : download URL
"""
custom_ops_mapper = {
    "libmmdeploy_onnxruntime_ops.so": "https://github.com/CatsWhoTrain/autoannotator/releases/download/0.0.1/libmmdeploy_onnxruntime_ops.so",
    "libonnxruntime.so.1.15.1": "https://github.com/CatsWhoTrain/autoannotator/releases/download/0.0.1/libonnxruntime.so.1.15.1",
}

custom_ops_storage_path = "custom_ops_storage"

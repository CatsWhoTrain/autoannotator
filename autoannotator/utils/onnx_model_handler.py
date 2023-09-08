import gc
import onnxruntime as ort
import onnx
from typing import Any, List
import numpy as np

from autoannotator.types.base import Device

ort.set_default_logger_severity(3)


class OnnxModelHandler:
    def __init__(self, model_path: str, device: str="cpu"):
        self.model_path = model_path
        self.start_session(device)
        
    def __call__(self, tensor: np.ndarray) -> Any:
        return self.forward(tensor)

    def start_session(self, device: str):
        self.onnx_model = onnx.load(self.model_path)
        onnx.checker.check_model(self.onnx_model)
        match device:
            case Device.CUDA:
                self.ort_sess = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
            case Device.CPU:
                self.ort_sess = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            case Device.RT:
                self.ort_sess = ort.InferenceSession(self.model_path, providers=['TensorrtExecutionProvider'])
            case _:
                raise Exception(f"Unknown device {device}")

    def stop_session(self):
        del self.ort_sess
        del self.onnx_model
        gc.collect()

    def get_input_shape(self) -> List[int]:
        shape = list()
        for val in self.onnx_model.graph.input[0].type.tensor_type.shape.dim[1:]:
            shape.append(val.dim_value)
        return shape

    def get_embedding_size(self) -> int:
        return self.onnx_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value

    def forward(self, x: np.ndarray):
        return self.ort_sess.run(None, {self.onnx_model.graph.input[0].name: x})[0]

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
from onnxruntime.quantization.preprocess import quant_pre_process
from PIL import Image


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OPTIMIZED_DIR = Path(__file__).resolve().parent.parent / "optimized_models"


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)


def download_model(model_name: str = "yolov8n") -> Path:
    from ultralytics import YOLO
    model = YOLO(f"{model_name}.pt")
    onnx_path = MODELS_DIR / f"{model_name}.onnx"
    model.export(format="onnx", imgsz=640, dynamic=False)
    src = Path(f"{model_name}.onnx")
    if src.exists():
        shutil.move(str(src), str(onnx_path))
    return onnx_path


def export_onnx(model_name: str = "yolov8n") -> Path:
    ensure_dirs()
    onnx_path = MODELS_DIR / f"{model_name}.onnx"
    if onnx_path.exists():
        print(f"[optimizer] ONNX model already exists: {onnx_path}")
        return onnx_path
    print(f"[optimizer] Exporting {model_name} to ONNX...")
    return download_model(model_name)


def _generate_calibration_data(num_samples: int = 32, img_size: int = 640):
    rng = np.random.default_rng(42)
    for i in range(num_samples):
        img = rng.integers(0, 256, size=(img_size, img_size, 3), dtype=np.uint8)
        yield np.expand_dims(img.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)


def quantize_onnx_static(
    onnx_path: Path,
    output_path: Path | None = None,
    calibration_samples: int = 32,
) -> Path:
    if output_path is None:
        output_path = OPTIMIZED_DIR / f"{onnx_path.stem}_int8.onnx"

    if output_path.exists():
        print(f"[optimizer] Quantized model already exists: {output_path}")
        return output_path

    print(f"[optimizer] Quantizing {onnx_path.name} -> {output_path.name} (static INT8)...")
    temp_dir = tempfile.mkdtemp()
    preprocessed = os.path.join(temp_dir, "preprocessed.onnx")
    quant_pre_process(str(onnx_path), preprocessed)

    calib_data = list(_generate_calibration_data(calibration_samples))
    from onnxruntime.quantization import CalibrationDataReader

    class YOLOCalibReader(CalibrationDataReader):
        def __init__(self, data):
            self.data = data
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.data):
                return None
            val = {"images": self.data[self.idx]}
            self.idx += 1
            return val

        def rewind(self):
            self.idx = 0

    quantize_static(
        preprocessed,
        str(output_path),
        calibration_data_reader=YOLOCalibReader(calib_data),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        extra_options={"ActivationSymmetric": True},
    )
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"[optimizer] Static INT8 quantization complete: {output_path}")
    return output_path


def quantize_onnx_dynamic(onnx_path: Path, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = OPTIMIZED_DIR / f"{onnx_path.stem}_int8_dynamic.onnx"

    if output_path.exists():
        print(f"[optimizer] Dynamic quantized model already exists: {output_path}")
        return output_path

    print(f"[optimizer] Quantizing {onnx_path.name} -> {output_path.name} (dynamic INT8)...")
    quantize_dynamic(str(onnx_path), str(output_path), weight_type=QuantType.QInt8)
    print(f"[optimizer] Dynamic INT8 quantization complete: {output_path}")
    return output_path


def get_model_size(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def print_comparison(fp32_path: Path, int8_path: Path):
    fp32_mb = get_model_size(fp32_path)
    int8_mb = get_model_size(int8_path)
    ratio = fp32_mb / int8_mb if int8_mb > 0 else 0
    print(f"\n[optimizer] Model size comparison:")
    print(f"  FP32: {fp32_mb:.2f} MB")
    print(f"  INT8: {int8_mb:.2f} MB")
    print(f"  Compression ratio: {ratio:.2f}x")

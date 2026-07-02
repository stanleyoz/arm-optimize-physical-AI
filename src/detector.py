from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class YOLODetector:
    def __init__(self, model_path: str | Path, conf_threshold: float = 0.4,
                 iou_threshold: float = 0.5, session_opts: ort.SessionOptions | None = None):
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if session_opts is None:
            session_opts = ort.SessionOptions()
            session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_opts.intra_op_num_threads = 4
            session_opts.inter_op_num_threads = 2

        self.session = ort.InferenceSession(
            self.model_path, sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = int(input_shape[2]) if len(input_shape) == 4 else 640
        self.input_width = int(input_shape[3]) if len(input_shape) == 4 else 640

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(self.input_width / w, self.input_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        x_offset = (self.input_width - new_w) // 2
        y_offset = (self.input_height - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return blob, scale, x_offset, y_offset

    def postprocess(
        self, outputs: list[np.ndarray], scale: float, x_offset: int, y_offset: int,
        frame_width: int, frame_height: int,
    ) -> list[dict]:
        predictions = np.squeeze(outputs[0]).T
        boxes, scores, class_ids = [], [], []

        for pred in predictions:
            confs = pred[4:]
            class_id = int(confs.argmax())
            score = float(confs[class_id])
            if score < self.conf_threshold:
                continue

            cx, cy, w, h = pred[:4]
            cx = (cx - x_offset) / scale
            cy = (cy - y_offset) / scale
            w = w / scale
            h = h / scale

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            x1 = max(0, min(x1, frame_width))
            y1 = max(0, min(y1, frame_height))
            x2 = max(0, min(x2, frame_width))
            y2 = max(0, min(y2, frame_height))

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        results = []
        for i in indices.flatten() if len(indices) > 0 else []:
            x1, y1, x2, y2 = boxes[i]
            results.append({
                "bbox": (x1, y1, x2, y2),
                "score": scores[i],
                "class_id": class_ids[i],
                "label": YOLO_CLASSES[class_ids[i]] if class_ids[i] < len(YOLO_CLASSES) else "unknown",
            })
        return results

    def infer(self, frame: np.ndarray) -> list[dict]:
        blob, scale, x_offset, y_offset = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        return self.postprocess(
            outputs, scale, x_offset, y_offset, frame.shape[1], frame.shape[0],
        )

    def warmup(self, iterations: int = 5):
        dummy = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.infer(dummy)


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} {det['score']:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame

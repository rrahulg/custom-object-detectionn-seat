import torch
from ultralytics import YOLO


def load_yolo_model(
    weights="yolov8s.pt", device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Loads a YOLO model.

    Args:
    - weights (str): Path to pretrained weights (default: "yolov8s.pt").
    - device (str): Device to load the model on ("cuda" or "cpu").

    Returns:
    - model (YOLO): Loaded YOLO model.
    """
    model = YOLO(weights).to(device)
    return model


def train_yolo_model(
    model,
    data_yaml="data/data.yml",
    epochs=50,
    batch_size=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Trains the YOLO model.

    Args:
    - model (YOLO): The YOLO model to train.
    - data_yaml (str): Path to dataset config (data.yml).
    - epochs (int): Number of training epochs.
    - batch_size (int): Training batch size.
    - device (str): Training device ("cuda" or "cpu").

    Returns:
    - None
    """
    model.train(data=data_yaml, epochs=epochs, batch=batch_size, device=device)


def validate_yolo_model(
    model,
    data_yaml="data/data.yml",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Validates the YOLO model.

    Args:
    - model (YOLO): The trained YOLO model.
    - data_yaml (str): Path to dataset config (data.yml).
    - device (str): Evaluation device ("cuda" or "cpu").

    Returns:
    - results: Validation results.
    """
    results = model.val(data=data_yaml, device=device)
    return results


def predict_yolo(model, image_path, conf_threshold=0.25):
    """
    Runs inference on an image using YOLO.

    Args:
    - model (YOLO): The YOLO model.
    - image_path (str): Path to the image for inference.
    - conf_threshold (float): Confidence threshold for predictions.

    Returns:
    - results: Inference results.
    """
    results = model(image_path, conf=conf_threshold)
    return results

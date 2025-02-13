from model.yolo_model import load_yolo_model, predict_yolo
import cv2

# Load trained YOLO model
model = load_yolo_model(weights="runs/train/exp/weights/best.pt")

# Path to input image
image_path = "test.jpg"

# Run inference
results = predict_yolo(model, image_path, conf_threshold=0.25)

# Save and show the annotated image
for result in results:
    annotated_image = result.plot()  # Get annotated image (OpenCV format)

    # Save annotated image
    output_path = "annotated_test.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved as {output_path}")

    # Show the image
    cv2.imshow("YOLO Detection", annotated_image)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()

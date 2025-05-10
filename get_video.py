import os
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from tqdm import tqdm
import numpy as np
from ufld.data.constant import culane_row_anchor

# Load class labels
def load_class_names():
    # GTSRB has 43 classes indexed from 0 to 42
    return [f"Class {i}" for i in range(43)]

# Visualize one frame with predictions
def visualize_prediction(image_tensor, boxes, labels, scores, threshold=0.5):
    keep = scores > threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    class_names = load_class_names()
    labels_text = [f"{class_names[label]}: {score:.2f}" for label, score in zip(labels, scores)]
    image_with_boxes = draw_bounding_boxes((image_tensor * 255).byte(), boxes, labels=labels_text, colors="red", width=2)
    return image_with_boxes

def run_video(model, frames_folder, output_video_path, device, threshold):
    # ----- Frame list (sort theo thứ tự tên file) -----
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))])
    frame_paths = [os.path.join(frames_folder, f) for f in frame_files]

    # ----- Init writer -----
    first_frame = cv2.imread(frame_paths[0])
    h, w, _ = first_frame.shape
    fps = 30  # đặt FPS tùy ý hoặc từ metadata nếu có
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

    lane_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
    ])

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
        lane_input = lane_transform(frame_rgb).unsqueeze(0).to(device)

        # ------------------- Predict lane -------------------
        with torch.no_grad():
            prediction = predict_lane(lane_input)

        lane_mask = create_lane_mask(prediction)
        lane_mask_resized = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_color = np.zeros((h, w, 3), dtype=np.uint8)
        mask_color[lane_mask_resized == 255] = [0, 0, 255]  # Red

        overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

        # ------------------- Predict traffic sign -------------------
        with torch.no_grad():
            outputs = model(input_tensor)[0]

        boxes = outputs['boxes'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()

        keep = scores > threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"{label}:{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Ghi frame đã overlay vào video
        out.write(overlay)

    out.release()

def run_video_file(detection_model, lane_model, input_video_path, output_video_path, device, threshold=0.5):
    # Anchor theo CULane
    col_sample = np.linspace(0, 800 - 1, 200)
    row_anchor = culane_row_anchor

    # Mở video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Cannot open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

    lane_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
    ])

    detection_model.eval()
    lane_model.eval()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            lane_input = lane_transform(frame_rgb).unsqueeze(0).to(device)

            # Lane prediction
            lane_output = lane_model(lane_input)[0]  # shape: [1, 201, 18, 4]
            lane_output = lane_output.data.cpu().numpy()
            lane_output = lane_output.transpose(1, 2, 0)  # [18, 4, 201]

            for i in range(4):  # 4 lanes
                for j in range(len(row_anchor)):
                    if np.max(lane_output[j, i, :-1]) <= 0:
                        continue
                    pos = np.argmax(lane_output[j, i, :-1])
                    if lane_output[j, i, -1] <= 0:
                        continue
                    x = int(col_sample[pos] * w / 800)
                    y = int(row_anchor[j] * h / 288)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Object detection
            outputs = detection_model(input_tensor)[0]
            boxes = outputs['boxes'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()

            keep = scores > threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}:{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved to: {output_video_path}")

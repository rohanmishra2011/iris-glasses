import cv2
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"   # change to "models/yolo11n.pt" if needed
CAMERA_INDEX = 0

THRESHOLDS = [0.40, 0.60, 0.80]
TRIALS_PER_THRESHOLD = 10
IMG_SIZE = 640

def normalize(label):
    label = label.lower().strip()
    aliases = {
        "cell phone": "phone",
        "mobile phone": "phone",
        "water bottle": "bottle",
        "remote control": "remote",
        "tv remote": "remote",
        "notebook": "book",
    }
    return aliases.get(label, label)

def parse_objects(text):
    return set(
        normalize(obj)
        for obj in text.split(",")
        if obj.strip()
    )

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Webcam did not open. Try CAMERA_INDEX = 1 or 2.")
    exit()

print("\nYOLO Live Threshold Test")
print("------------------------")
print("This tests 10 IMAGES per confidence threshold.")
print("Each image can contain multiple objects.")
print("For each image:")
print("1. Put the scene in front of the camera.")
print("2. Press SPACE to capture.")
print("3. Type every real object visible, separated by commas.")
print("Example: phone,bottle,book")
print("Press Q to quit early.\n")

all_results = []
csv_rows = []

for threshold in THRESHOLDS:
    print(f"\n==============================")
    print(f"Testing confidence threshold: {threshold}")
    print(f"==============================")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    trial = 1

    while trial <= TRIALS_PER_THRESHOLD:
        ok, frame = cap.read()

        if not ok:
            print("Could not read webcam frame.")
            break

        result = model.predict(
            frame,
            conf=threshold,
            imgsz=IMG_SIZE,
            verbose=False
        )[0]

        annotated = result.plot()

        cv2.putText(
            annotated,
            f"Confidence: {threshold} | Image {trial}/{TRIALS_PER_THRESHOLD}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            "SPACE = capture image | Q = quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Live YOLO Threshold Test", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quit early.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        if key == 32:  # SPACE
            detected_objects = []

            for class_id in result.boxes.cls.tolist():
                label = result.names[int(class_id)]
                detected_objects.append(normalize(label))

            detected_set = set(detected_objects)

            print(f"\nThreshold {threshold}, Image {trial}")
            print(f"YOLO detected: {sorted(detected_set)}")

            actual_input = input(
                "Enter every real object visible, separated by commas: "
            )

            actual_set = parse_objects(actual_input)

            true_positives = actual_set.intersection(detected_set)
            false_positives = detected_set - actual_set
            false_negatives = actual_set - detected_set

            tp = len(true_positives)
            fp = len(false_positives)
            fn = len(false_negatives)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            print(f"Actual objects: {sorted(actual_set)}")
            print(f"Correct detections: {sorted(true_positives)}")
            print(f"False positives: {sorted(false_positives)}")
            print(f"Missed objects: {sorted(false_negatives)}")
            print(f"TP={tp}, FP={fp}, FN={fn}")

            csv_rows.append({
                "threshold": threshold,
                "image_trial": trial,
                "actual_objects": ", ".join(sorted(actual_set)),
                "detected_objects": ", ".join(sorted(detected_set)),
                "true_positives": ", ".join(sorted(true_positives)),
                "false_positives": ", ".join(sorted(false_positives)),
                "false_negatives": ", ".join(sorted(false_negatives)),
                "tp": tp,
                "fp": fp,
                "fn": fn
            })

            trial += 1
            time.sleep(0.5)

    precision, recall = calculate_metrics(total_tp, total_fp, total_fn)

    all_results.append({
        "threshold": threshold,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall
    })

cap.release()
cv2.destroyAllWindows()

print("\n\nFINAL RESULTS")
print("-------------")

for result in all_results:
    print(f"\nConfidence: {result['threshold']}")
    print(f"TP: {result['tp']}")
    print(f"FP: {result['fp']}")
    print(f"FN: {result['fn']}")
    print(f"Precision: {result['precision'] * 100:.1f}%")
    print(f"Recall: {result['recall'] * 100:.1f}%")

with open("yolo_threshold_test_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)

threshold_labels = [str(r["threshold"]) for r in all_results]
precision_values = [r["precision"] * 100 for r in all_results]
recall_values = [r["recall"] * 100 for r in all_results]

x = np.arange(len(threshold_labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, precision_values, width, label="Precision", color="#174A6A")
plt.bar(x + width / 2, recall_values, width, label="Recall", color="#2E7D57")

plt.title("YOLO Precision and Recall at Different Confidence Thresholds")
plt.xlabel("Confidence Threshold")
plt.ylabel("Percentage (%)")
plt.xticks(x, threshold_labels)
plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.25)
plt.legend()

for i in range(len(threshold_labels)):
    plt.text(x[i] - width / 2, precision_values[i] + 1, f"{precision_values[i]:.0f}%", ha="center")
    plt.text(x[i] + width / 2, recall_values[i] + 1, f"{recall_values[i]:.0f}%", ha="center")

plt.tight_layout()
plt.savefig("yolo_threshold_precision_recall.png", dpi=300)
plt.show()

print("\nSaved files:")
print("yolo_threshold_precision_recall.png")
print("yolo_threshold_test_results.csv")
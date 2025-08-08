import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# === Part 3: Face detection ===
# Load lại mô hình
model = joblib.load("ai/lesson2/face_detect/svm_model.pkl")
W = model['W']
b = model['b']

# Config detection
img_path = "ai/lesson2/docs/img1.jpg"
conf_thresh = 0.5
nms_thresh = 0.2

# Load và chuyển grayscale
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W_img = gray.shape

# Trượt cửa sổ 24x24 qua ảnh
step = 4
patches = []
boxes = []

for y in range(0, H - 24 + 1, step):
    for x in range(0, W_img - 24 + 1, step):
        patch = gray[y:y+24, x:x+24].flatten()
        patches.append(patch)
        boxes.append([x, y, x+24, y+24])

patches = np.array(patches)
boxes = np.array(boxes)

# show image with patches
# for box in boxes:
#     x1, y1, x2, y2 = box
#     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title("Image with patches")
# plt.show()

# # show image with boxes
# for box in boxes:
#     x1, y1, x2, y2 = box
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title("Image with boxes")
# plt.show()

# Tính confidence cho từng patch
confidences = patches @ W + b

print(f"Total patches: {len(patches)}")

# Lọc những patch có confidence > threshold
keep = confidences > conf_thresh
boxes_keep = boxes[keep]
conf_keep = confidences[keep]

print(f"keep {len(boxes_keep)} boxes with confidence > {conf_thresh:.2f}")
print(f"boxes_keep length: {len(boxes_keep)}")
print(f"conf_keep length: {len(conf_keep)}")


# NMS
indices = cv2.dnn.NMSBoxes(
    bboxes=boxes_keep.tolist(),
    scores=conf_keep.tolist(),
    score_threshold=conf_thresh,
    nms_threshold=nms_thresh
)

# Vẽ kết quả
face_count = len(indices)
if len(indices) > 0:
    for i, idx in enumerate(indices):
        idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x1, y1, x2, y2 = boxes_keep[idx]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img, f"{idx+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
else:
    print("⚠️ No boxes passed NMS.")

cv2.imwrite("ai/lesson2/face_detect/detections_nms.png", img)
# open the image to see results using plot
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
print("✅ Detection result saved to detections_nms.png", f"with {face_count} faces detected.")



# === Visualization: draw all boxes with confidence text at center ===
scale = 10
vis_img = cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = (np.array(box) * scale).astype(int)
    conf = float(confidences[i])

    # Vẽ khung
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (160, 160, 160), 1)

    # Text ở giữa
    text = f"{conf:.1f}"
    cv2.putText(vis_img, text, (x1+10, y1+24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (225, 225, 225), 1, lineType=cv2.LINE_AA)

vis_path = "ai/lesson2/face_detect/confidences_text_boxes_scaled.png"
cv2.imwrite(vis_path, vis_img)
# plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title("Confidence Visualization (Centered Text)")
# plt.show()
print(f"📌 Saved confidence visualization (centered text) to: {vis_path}")

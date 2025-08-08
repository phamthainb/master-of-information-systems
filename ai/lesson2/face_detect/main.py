from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time
import joblib

# Load positive and negative sample matrices
pos_data = loadmat('ai/lesson2/docs/possamples.mat')
neg_data = loadmat('ai/lesson2/docs/negsamples.mat')

# In thử keys trong file để kiểm tra tên biến thực tế
# print("Positive sample keys:", pos_data.keys())
# print("Negative sample keys:", neg_data.keys())

# print("Positive sample keys:", pos_data["possamples"].shape)
# print("Negative sample keys:", neg_data["negsamples"].shape)

# for i in range(10):
#     plt.imshow(neg_data['negsamples'][:, :, i])
#     plt.show()

pos_mat = pos_data['possamples']
neg_mat = neg_data['negsamples']

## Part 1. 
# Flatten dữ liệu
X_pos = pos_mat.reshape(24*24, -1).T  # shape (4000, 576)
X_neg = neg_mat.reshape(24*24, -1).T  # shape (17256, 576)

# Tạo nhãn: 1 cho pos, 0 cho neg
y_pos = np.ones(X_pos.shape[0])
y_neg = np.zeros(X_neg.shape[0])

# Ghép pos + neg
X = np.vstack((X_pos, X_neg))
y = np.concatenate((y_pos, y_neg))

# Tách train / val
# test 20% dữ liệu
Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", Xtrain.shape, ytrain.shape)
print("Val shape:", Xval.shape, yval.shape)

# === Part 2: SVM classification ===
# add more regularization
C_list = [0.001, 0.01, 0.1, 1, 10, 100]

best_acc = 0
best_W = None
best_b = None
best_C = None

for C in C_list:
    print(f"\n==> Training LinearSVC with C={C} ...")
    start_time = time.time()

    clf = LinearSVC(C=C, max_iter=10000)
    clf.fit(Xtrain, ytrain)

    elapsed = time.time() - start_time
    W = clf.coef_.flatten()       # shape (576,)
    b = clf.intercept_[0]

    # Tính lại confidence thủ công
    train_conf = Xtrain @ W + b
    val_conf = Xval @ W + b

    # Dự đoán
    y_train_pred = (train_conf >= 0).astype(int)
    y_val_pred = (val_conf >= 0).astype(int)

    # Accuracy thủ công
    acc_train = accuracy_score(ytrain, y_train_pred)
    acc_val = accuracy_score(yval, y_val_pred)

    print(f"C={C} | Train Acc = {acc_train:.4f} | Val Acc = {acc_val:.4f} | Time: {elapsed:.2f}s")

    # Lưu mô hình tốt nhất theo validation accuracy
    if acc_val > best_acc:
        best_acc = acc_val
        best_W = W
        best_b = b
        best_C = C

    # Hiển thị W dưới dạng ảnh
    plt.imshow(W.reshape(24, 24), cmap='gray')
    plt.title(f"W as image (C={C})")
    plt.colorbar()
    # plt.show()
    plt.savefig(f"ai/lesson2/face_detect/W_image_C_{C}.png")
    plt.close()

# In kết quả tốt nhất
print(f"\nBest C = {best_C} with Val Accuracy = {best_acc:.4f}")


# Lưu model tốt nhất
model_path = "ai/lesson2/face_detect/svm_model.pkl"
joblib.dump({'W': best_W, 'b': best_b, 'C': best_C}, model_path)
print(f"Saved model to {model_path}")

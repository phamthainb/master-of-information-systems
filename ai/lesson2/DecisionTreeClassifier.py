import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Dữ liệu huấn luyện (theo bảng)
train_df = pd.DataFrame({
    'time': ['1-2', '2-7', '>7', '1-2', '>7', '1-2', '2-7', '2-7'],
    'gender': ['m', 'm', 'f', 'f', 'm', 'm', 'f', 'm'],
    'area': ['urban', 'rural', 'rural', 'rural', 'rural', 'rural', 'urban', 'urban'],
    'risk': ['low', 'high', 'low', 'high', 'high', 'high', 'low', 'low']
})

# Dữ liệu cần dự đoán
test_df = pd.DataFrame({
    'time': ['1-2', '2-7', '1-2'],
    'gender': ['f', 'm', 'f'],
    'area': ['rural', 'urban', 'urban']
}, index=['A', 'B', 'C'])

# Encode thủ công bằng LabelEncoder
encoders = {}
for column in ['time', 'gender', 'area', 'risk']:
    le = LabelEncoder()
    le.fit(train_df[column])
    train_df[column] = le.transform(train_df[column])
    encoders[column] = le

# Áp dụng encoder cho test set
for column in ['time', 'gender', 'area']:
    test_df[column] = encoders[column].transform(test_df[column])

# Huấn luyện mô hình Decision Tree
X_train = train_df[['time', 'gender', 'area']]
y_train = train_df['risk']

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# Dự đoán cho test set
X_test = test_df[['time', 'gender', 'area']]
predictions = clf.predict(X_test)

# In kết quả (decode label)
for idx, pred in zip(test_df.index, predictions):
    print(f"ID {idx} => Risk: {encoders['risk'].inverse_transform([pred])[0]}")
    # print more details if needed
    print(f"Probabilities: {pred}")

# Vẽ cây quyết định
# plt.figure(figsize=(10, 6))
# plot_tree(clf, feature_names=['time', 'gender', 'area'], class_names=encoders['risk'].classes_, filled=True)
# plt.title("Decision Tree for Risk Prediction")
# plt.show()

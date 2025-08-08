from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
import pandas as pd

# Dữ liệu huấn luyện từ bảng
data = [
    ['<=30', 'high',    'no',  'fair',      'no'],
    ['<=30', 'high',    'no',  'excellent', 'no'],
    ['31...40', 'high', 'no',  'fair',      'yes'],
    ['>40',  'medium',  'no',  'fair',      'yes'],
    ['>40',  'low',     'yes', 'fair',      'yes'],
    ['>40',  'low',     'yes', 'excellent', 'no'],
    ['31...40', 'low',  'yes', 'excellent', 'yes'],
    ['<=30', 'medium',  'no',  'fair',      'no'],
    ['<=30', 'low',     'yes', 'fair',      'yes'],
    ['>40',  'medium',  'yes', 'fair',      'yes'],
    ['<=30', 'medium',  'yes', 'excellent', 'yes'],
    ['31...40', 'medium','no', 'excellent', 'yes'],
    ['31...40', 'high', 'yes', 'fair',      'yes'],
    ['>40',  'medium',  'no',  'excellent', 'no'],
]

columns = ['age', 'income', 'student', 'credit_rating', 'buys_computer']
df = pd.DataFrame(data, columns=columns)

# Encode categorical features
le = {col: LabelEncoder() for col in columns}
for col in columns:
    df[col] = le[col].fit_transform(df[col])

# Tách features và label
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Train Naive Bayes
model = CategoricalNB()
model.fit(X, y)

# Dự đoán cho input mới:
# Age <=30, income=medium, student=yes, credit_rating=fair
new_sample = [['<=30', 'medium', 'yes', 'fair']]
new_sample_encoded = [le[col].transform([val])[0] for col, val in zip(X.columns, new_sample[0])]

# Dự đoán và giải mã kết quả
pred = model.predict([new_sample_encoded])
result = le['buys_computer'].inverse_transform(pred)[0]

print("Prediction:", result)

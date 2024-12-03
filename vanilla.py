from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data, mnist.target.astype(int)

# Filter for binary classification
binary_classes = [3, 8]
X, y = X[np.isin(y, binary_classes)], y[np.isin(y, binary_classes)]

# Convert labels to 0/1
y = np.where(y == binary_classes[0], 0, 1)

# Resize images
def downsample(img, factor=2):
    size = int(np.sqrt(img.shape[0]))
    small_size = size // factor
    img_reshaped = img.reshape(size, size)
    small_img = img_reshaped.reshape(small_size, factor, small_size, factor).mean(axis=(1, 3))
    return small_img.flatten()

X_resized = np.apply_along_axis(downsample, 1, X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Plaintext Accuracy:", accuracy_score(y_test, y_pred))
print("Plaintext AUROC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

from openfhe import *
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import time

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

# Train a plaintext logistic regression model for comparison
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate on plaintext data
y_pred = model.predict(X_test)
print("Plaintext Accuracy:", accuracy_score(y_test, y_pred))
print("Plaintext AUROC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Set up OpenFHE for encryption
params = CCParamsCKKSRNS()  # CKKS scheme for approximate HE
params.SetMultiplicativeDepth(15)  # Depth for gradient descent iterations
params.SetScalingModSize(40)       # Scaling modulus size for precision
context = GenCryptoContext(params)

# Enable necessary features
context.Enable(PKESchemeFeature.PKE)
context.Enable(PKESchemeFeature.KEYSWITCH)
context.Enable(PKESchemeFeature.LEVELEDSHE)
context.Enable(PKESchemeFeature.ADVANCEDSHE)

# Key generation
keypair = context.KeyGen()
context.EvalMultKeyGen(keypair.secretKey)

# Encrypt the training data and labels
subset_size = 180  # Smaller dataset for demonstration
X_train_small = X_train[:subset_size]
y_train_small = y_train[:subset_size]

# Encrypt training data
ptxt_list = [context.MakeCKKSPackedPlaintext(row.tolist()) for row in X_train_small]
X_train_encrypted = [context.Encrypt(keypair.publicKey, ptxt) for ptxt in ptxt_list]

# Encrypt labels
y_train_encrypted = [context.Encrypt(keypair.publicKey, context.MakeCKKSPackedPlaintext([y]))
                     for y in y_train_small]

# Initialize weights
initial_weights = context.Encrypt(keypair.publicKey, context.MakeCKKSPackedPlaintext([0.0] * X_train.shape[1]))

# Learning rate
learning_rate = context.MakeCKKSPackedPlaintext([1.0])

# Polynomial approximation of sigmoid function
def polynomial_sigmoid(x, context):
    const_0_5 = context.MakeCKKSPackedPlaintext([0.5])
    const_neg_0_0843 = context.MakeCKKSPackedPlaintext([-0.0843])
    const_0_0002 = context.MakeCKKSPackedPlaintext([0.0002])

    const_0_5_cipher = context.Encrypt(keypair.publicKey, const_0_5)
    const_neg_0_0843_cipher = context.Encrypt(keypair.publicKey, const_neg_0_0843)
    const_0_0002_cipher = context.Encrypt(keypair.publicKey, const_0_0002)


    x_squared = context.EvalMult(x, x)
    x_squared = context.ModReduce(x_squared)
    
    x_cubed = context.EvalMult(x_squared, x)
    x_cubed = context.ModReduce(x_cubed)

    term1 = context.EvalMult(x, const_neg_0_0843_cipher)
    term1 = context.ModReduce(term1)
    
    term2 = context.EvalMult(x_cubed, const_0_0002_cipher)
    term2 = context.ModReduce(term2)

    result = context.EvalAdd(term1, const_0_5_cipher)
    result = context.EvalAdd(result, term2)

    return result

# One iteration of encrypted gradient descent
def gradient_descent_step(X_encrypted, y_encrypted, weights, learning_rate, context):
    # Compute predictions: sigmoid(X @ weights)
    predictions = [context.EvalMult(X_row, weights) for X_row in X_encrypted]
    predictions = [polynomial_sigmoid(p, context) for p in predictions]

    # Compute error: predictions - y
    errors = [context.EvalSub(pred, y) for pred, y in zip(predictions, y_encrypted)]

    # Compute gradient: dot product of errors with features
    gradients = [context.EvalMult(err, X_row) for err, X_row in zip(errors, X_encrypted)]

    # Sum gradients across all rows to get the overall gradient
    total_gradient = gradients[0]
    for grad in gradients[1:]:
        total_gradient = context.EvalAdd(total_gradient, grad)

    # Update weights: w = w - alpha * gradient
    updated_weights = context.EvalSub(weights, context.EvalMult(learning_rate, total_gradient))

    return updated_weights


# Perform gradient descent
num_iterations = 2
weights = initial_weights

for i in range(num_iterations):
    print(f"Starting iteration {i+1}...")
    start = time.perf_counter()
    weights = gradient_descent_step(X_train_encrypted, y_train_encrypted, weights, learning_rate, context)
    end = time.perf_counter()
    print(f"Iteration {i+1} completed. ({end - start:.2f}s)")

# Decrypt and evaluate the model
decrypted_weights = context.Decrypt(keypair.secretKey, weights).GetCKKSPackedValue()

# Encrypt test data
X_test_encrypted = [context.Encrypt(keypair.publicKey, context.MakeCKKSPackedPlaintext(row.tolist()))
                    for row in X_test[:subset_size]]

# Compute predictions on encrypted test data
test_predictions = [context.EvalMult(X_row, weights) for X_row in X_test_encrypted]
test_predictions = [polynomial_sigmoid(p, context) for p in test_predictions]

# Decrypt predictions
decrypted_predictions = [context.Decrypt(keypair.secretKey, pred).GetCKKSPackedValue()[0]
                         for pred in test_predictions]

# Binarize predictions
y_test_pred = np.array(decrypted_predictions) > 0.5

# Evaluate accuracy
print("Encrypted Test Accuracy:", accuracy_score(y_test[:subset_size], y_test_pred))


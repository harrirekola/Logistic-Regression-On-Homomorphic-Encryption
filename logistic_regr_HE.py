from openfhe import *
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import time

def batch_encrypt(data, context, public_key, batch_size):
    """
    Encrypt data in batches.
    
    Args:
        data (list or np.ndarray): List or array of vectors (e.g., training data or labels).
        context: OpenFHE context.
        public_key: Encryption public key.
        batch_size (int): Number of samples to pack into one batch.
    
    Returns:
        list: List of encrypted ciphertexts.
    """
    encrypted_batches = []
    if len(data.shape) == 1:  # If the data is 1D (e.g., labels)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].tolist()
            ptxt = context.MakeCKKSPackedPlaintext(batch)
            encrypted_batches.append(context.Encrypt(public_key, ptxt))
    else:  # If the data is 2D (e.g., training features)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            max_len = batch.shape[1]  # Assuming all rows are the same size
            # Flatten the 2D batch into a 1D list
            flat_batch = batch.flatten().tolist()
            ptxt = context.MakeCKKSPackedPlaintext(flat_batch)
            encrypted_batches.append(context.Encrypt(public_key, ptxt))
    return encrypted_batches



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
subset_size = 120  # Smaller dataset for demonstration
X_train_small = X_train[:1000]
y_train_small = y_train[:200]

# Encrypt training data
batch_size = 30
X_train_encrypted = batch_encrypt(X_train_small, context, keypair.publicKey, batch_size)
#ptxt_list = [context.MakeCKKSPackedPlaintext(row.tolist()) for row in X_train_small]
#X_train_encrypted = [context.Encrypt(keypair.publicKey, ptxt) for ptxt in ptxt_list]

# Encrypt labels
y_train_encrypted = batch_encrypt(y_train_small, context, keypair.publicKey, batch_size)
#y_train_encrypted = [context.Encrypt(keypair.publicKey, context.MakeCKKSPackedPlaintext([y]))
#                     for y in y_train_small]

# Initialize weights
initial_weights = context.Encrypt(keypair.publicKey, context.MakeCKKSPackedPlaintext([0.0] * X_train.shape[1]))

# Learning rate
learning_rate = context.MakeCKKSPackedPlaintext([1.0])

""""
def polynomial_sigmoid(x, context):
    const_0_5 = context.MakeCKKSPackedPlaintext([0.5])
    const_neg_0_0843 = context.MakeCKKSPackedPlaintext([-0.0843])
    const_0_0002 = context.MakeCKKSPackedPlaintext([0.0002])

    const_0_5_cipher = context.Encrypt(keypair.publicKey, const_0_5)
    const_neg_0_0843_cipher = context.Encrypt(keypair.publicKey, const_neg_0_0843)
    const_0_0002_cipher = context.Encrypt(keypair.publicKey, const_0_0002)

    x_squared = context.EvalMult(x, x)
    x_cubed = context.EvalMult(x_squared, x)

    term1 = context.EvalMult(const_neg_0_0843_cipher, x)
    term2 = context.EvalMult(const_0_0002_cipher, x_cubed)

    result = context.EvalAdd(const_0_5_cipher, term1)
    result = context.EvalAdd(result, term2)

    return result
"""

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


# Gradient descent iteration step
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
print("Performing gradient descent...")
num_iterations = 1
weights = initial_weights

for i in range(num_iterations):
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


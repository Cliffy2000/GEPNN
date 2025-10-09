import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from core.network import Network

# Cache for dataset
_dataset_cache = {}


def get_iris_data():
    """Load and prepare Iris dataset."""
    if 'iris' not in _dataset_cache:
        # Load dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y
        )

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        _dataset_cache['iris'] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    return _dataset_cache['iris']


def evaluate_iris(individual):
    data = get_iris_data()
    X_train = data['X_train']
    y_train = data['y_train']

    # Create network
    try:
        network = Network(individual)
    except:
        return (0.0,)

    predictions = []

    # Evaluate each sample
    for i in range(len(X_train)):
        # Create input dict
        inputs = {f'x{j}': X_train[i, j] for j in range(4)}

        try:
            for n in network.nodes:
                n.prev_value = 0.0
            # Get network output
            output = network.forward(inputs)
            output_val = output.item() if torch.is_tensor(output) else float(output)

            # Map to class (3 classes: 0, 1, 2)
            if output_val < -0.33:
                pred = 0
            elif output_val < 0.33:
                pred = 1
            else:
                pred = 2

            predictions.append(output_val)

        except:
            predictions.append(0)  # Default prediction on error

    # Calculate accuracy
    accuracy = accuracy_score(y_train, predictions)

    return (accuracy,)

'''
def evaluate_xor(individual):
    """
    Evaluate individual on XOR classification task using hybrid fitness.

    Args:
        individual: GEP individual to evaluate

    Returns:
        Tuple with hybrid fitness score (combines MSE and accuracy)
    """
    # Define the 4 XOR patterns
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR outputs

    n_samples = 4

    # Create network
    try:
        network = Network(individual)
    except:
        return (0.0,)

    correct = 0
    total_squared_error = 0.0

    # Evaluate each sample
    for i in range(n_samples):
        # Create input dict
        inputs = {
            'x0': float(X[i, 0]),
            'x1': float(X[i, 1])
        }

        try:
            # Get network output
            output = network.forward(inputs)
            output_val = output.item() if torch.is_tensor(output) else float(output)

            # Clamp output to avoid extreme values
            output_val = max(0.001, min(0.999, output_val))

            # Calculate squared error
            squared_error = (y[i] - output_val) ** 2
            total_squared_error += squared_error

            # Threshold at 0.5 for binary classification
            pred = 1 if output_val > 0.5 else 0

            # Check if correct
            if pred == y[i]:
                correct += 1

        except:
            # On error, add maximum squared error (1.0)
            total_squared_error += 1.0

    # Calculate components
    mse = total_squared_error / n_samples
    accuracy = correct / n_samples

    # Hybrid fitness: weighted combination of MSE and accuracy
    # Higher weight on MSE (0.7) to provide more granular feedback
    w_mse = 0.7
    w_acc = 0.3

    fitness = w_mse * (1 - mse) + w_acc * accuracy

    return (fitness,)
'''


def evaluate_xor_prev(individual):
    """
    Evaluate individual on XOR classification task using hybrid fitness.

    Args:
        individual: GEP individual to evaluate

    Returns:
        Tuple with hybrid fitness score (combines MSE and accuracy)
    """
    # Define the 4 XOR patterns
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR outputs

    # Randomize the order
    indices = np.random.permutation(4)

    # Create network
    try:
        network = Network(individual)
    except:
        return (0.0,)

    correct = 0
    total_squared_error = 0.0

    # Evaluate each sample in random order
    for i in indices:
        # Create input dict
        inputs = {
            'x0': float(X[i, 0]),
            'x1': float(X[i, 1])
        }

        try:
            for n in network.nodes:
                n.prev_value = 0.0
            # Get network output
            output = network.forward(inputs)
            output_val = output.item() if torch.is_tensor(output) else float(output)

            # Clamp output to avoid extreme values
            output_val = max(0.001, min(0.999, output_val))

            # Calculate squared error
            squared_error = (y[i] - output_val) ** 2
            total_squared_error += squared_error

            # Threshold at 0.5 for binary classification
            pred = 1 if output_val > 0.5 else 0

            # Check if correct
            if pred == y[i]:
                correct += 1

        except:
            # On error, add maximum squared error (1.0)
            total_squared_error += 1.0

    # Calculate components
    mse = total_squared_error / 4
    accuracy = correct / 4

    # Hybrid fitness: weighted combination of MSE and accuracy
    # Higher weight on MSE (0.7) to provide more granular feedback
    w_mse = 0.7
    w_acc = 0.3

    fitness = (w_mse * (1 - mse) + w_acc * accuracy)

    return (fitness,)


def evaluate_xor(individual):
    """
    Evaluate individual on XOR classification task using hybrid fitness.

    Args:
        individual: GEP individual to evaluate

    Returns:
        Tuple with hybrid fitness score (combines MSE and accuracy)
    """
    # Define the 4 XOR patterns as batch
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)

    # Create network
    try:
        network = Network(individual)
    except:
        return (0.0,)

    try:
        # Process all 4 patterns at once
        # X shape: (4, 2) - 4 samples, 2 inputs
        outputs = network.forward(X)  # Returns shape (4,) - one output per sample

        # Clamp outputs to avoid extreme values
        outputs = np.clip(outputs, 0.001, 0.999)

        # Calculate MSE
        squared_errors = (y - outputs) ** 2
        mse = np.mean(squared_errors)

        # Calculate accuracy
        predictions = (outputs > 0.5).astype(int)
        accuracy = np.mean(predictions == y)

    except:
        # On error, worst case fitness
        return (0.0,)

    # Hybrid fitness: weighted combination of MSE and accuracy
    w_mse = 0.7
    w_acc = 0.3
    fitness = w_mse * (1 - mse) + w_acc * accuracy

    return (fitness,)


def evaluate_txor(individual):
    """
    Evaluate individual on Temporal XOR with k=2.
    Output at time t is XOR of inputs at t-1 and t-2.

    Args:
        individual: GEP individual to evaluate

    Returns:
        Tuple with hybrid fitness score
    """
    # Parameters
    batch_size = 100
    seq_length = 10
    k = 2

    # Generate random binary sequences (as integers, then convert)
    sequences = np.random.randint(0, 2, size=(batch_size, seq_length)).astype(np.float32)

    # Create network
    try:
        network = Network(individual)
    except:
        return (0.0,)

    try:
        # Reset network state for new sequences
        network.prev_values = None

        total_squared_error = 0
        correct = 0
        total_predictions = 0

        # Process each timestep
        for t in range(seq_length):
            # Input at current timestep - shape (batch_size, 1)
            inputs = sequences[:, t:t + 1]

            # Forward pass
            outputs = network.forward(inputs)  # Shape (batch_size,)
            outputs = np.clip(outputs, 0.001, 0.999)

            # Calculate targets for t >= k
            if t >= k:
                # XOR of t-1 and t-2
                targets = np.logical_xor(sequences[:, t - 1], sequences[:, t - 2]).astype(np.float32)

                # MSE
                squared_errors = (targets - outputs) ** 2
                total_squared_error += np.sum(squared_errors)

                # Accuracy
                predictions = (outputs > 0.5).astype(int)
                correct += np.sum(predictions == targets)

                total_predictions += batch_size

        # Calculate metrics
        mse = total_squared_error / total_predictions if total_predictions > 0 else 1.0
        accuracy = correct / total_predictions if total_predictions > 0 else 0.0

        # Hybrid fitness
        w_mse = 0.7
        w_acc = 0.3
        fitness = w_mse * (1 - mse) + w_acc * accuracy

    except Exception as e:
        print(f"Evaluation error: {e}")
        return (0.0,)

    return (fitness,)


def evaluate_binary_counter(individual):
    """
    Evaluate individual on binary counting task.
    Output at time t is the count of 1s seen so far, represented in binary.
    """
    batch_size = 200
    seq_length = 10

    # Generate random binary sequences
    sequences = np.random.randint(0, 2, size=(batch_size, seq_length)).astype(np.float32)

    # Create network
    try:
        network = Network(individual)
    except:
        return (0.0,)

    try:
        # Reset network state
        network.prev_values = None

        total_squared_error = 0
        correct = 0
        total_predictions = 0

        # Track running count for each sequence in batch
        counts = np.zeros(batch_size, dtype=np.float32)

        # Process each timestep
        for t in range(seq_length):
            # Input at current timestep - shape (batch_size, 1)
            inputs = sequences[:, t:t + 1]

            # Forward pass
            outputs = network.forward(inputs)  # Shape (batch_size,)

            # Target is the current count in binary (0 or 1 for LSB)
            # For simplicity, output the LSB of the count
            targets = np.mod(counts, 2).astype(np.float32)

            # Clip outputs
            outputs = np.clip(outputs, 0.001, 0.999)

            # Calculate MSE
            squared_errors = (targets - outputs) ** 2
            total_squared_error += np.sum(squared_errors)

            # Calculate accuracy
            predictions = (outputs > 0.5).astype(int)
            correct += np.sum(predictions == targets)
            total_predictions += batch_size

            # Update counts for next timestep
            counts += sequences[:, t]

        # Calculate metrics
        mse = total_squared_error / total_predictions
        accuracy = correct / total_predictions

        # Hybrid fitness
        w_mse = 0.2
        w_acc = 0.8
        fitness = w_mse * (1 - mse) + w_acc * accuracy

    except Exception as e:
        print(f"Evaluation error: {e}")
        return (0.0,)

    return (fitness,)
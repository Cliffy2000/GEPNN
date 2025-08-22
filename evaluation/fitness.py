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
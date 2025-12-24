import numpy as np
from core.network import Network

_iris_cache = None


def get_iris_data():
    """Load and prepare Iris dataset for quick sanity checks."""
    global _iris_cache

    if _iris_cache is None:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        iris = load_iris()
        X, y = iris.data, iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        _iris_cache = {
            'X_train': X_train.astype(np.float32),
            'X_test': X_test.astype(np.float32),
            'y_train': y_train,
            'y_test': y_test
        }

    return _iris_cache


def evaluate_iris(individual):
    """
    Evaluate individual on Iris classification task.
    Used as a quick sanity check for network functionality.
    """
    data = get_iris_data()
    X_train = data['X_train']
    y_train = data['y_train']

    try:
        network = Network(individual)
    except:
        return (0.0,)

    try:
        network.prev_values = None
        outputs = network.forward(X_train)
        outputs = np.clip(outputs, -1.0, 1.0)

        # Map continuous output to 3 classes
        predictions = np.zeros(len(outputs), dtype=int)
        predictions[outputs >= 0.33] = 2
        predictions[(outputs >= -0.33) & (outputs < 0.33)] = 1
        predictions[outputs < -0.33] = 0

        correct = np.sum(predictions == y_train)
        accuracy = correct / len(y_train)

    except:
        return (0.0,)

    return (accuracy,)


def evaluate_xor(individual):
    """
    Evaluate individual on XOR classification task.

    Fitness: 50/50 weighted combination of normalized MSE and accuracy.
    MSE component normalized with factor of 4 (max MSE for binary = 0.25).
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)

    try:
        network = Network(individual)
    except:
        return (0.0,)

    try:
        network.prev_values = None
        outputs = network.forward(X)
        outputs = np.clip(outputs, 0.001, 0.999)

        # MSE component (normalized)
        mse = np.mean((y - outputs) ** 2)
        mse_component = max(0.0, 1.0 - 4.0 * mse)

        # Accuracy component
        predictions = (outputs > 0.5).astype(int)
        accuracy = np.mean(predictions == y)

        fitness = 0.5 * mse_component + 0.5 * accuracy

    except:
        return (0.0,)

    return (fitness,)


def evaluate_txor(individual, t1=-1, t2=-2):
    """
    Evaluate individual on Temporal XOR task.

    Output at time t is XOR of inputs at (t + t1) and (t + t2).
    For example:
        t1=-1, t2=-2: XOR(x[t-1], x[t-2])  -- standard T-XOR k=2
        t1=0,  t2=-1: XOR(x[t], x[t-1])    -- T-XOR k=1
        t1=0,  t2=-2: XOR(x[t], x[t-2])    -- mixed

    Args:
        individual: GEP individual to evaluate
        t1: first timestep offset (0 or negative)
        t2: second timestep offset (0 or negative)

    Returns:
        Tuple with fitness score
    """
    batch_size = 500
    seq_length = 15

    # Determine k (number of timesteps to wait before evaluating)
    k = max(abs(t1), abs(t2))

    rng = np.random.default_rng(seed=42)
    sequences = rng.integers(0, 2, size=(batch_size, seq_length)).astype(np.float32)

    try:
        network = Network(individual)
    except:
        return (0.0,)

    try:
        network.prev_values = None

        total_squared_error = 0.0
        correct = 0
        total_predictions = 0

        for t in range(seq_length):
            inputs = sequences[:, t:t + 1]
            outputs = network.forward(inputs)
            outputs = np.clip(outputs, 0.001, 0.999)

            if t >= k:
                # XOR of inputs at (t + t1) and (t + t2)
                targets = np.logical_xor(
                    sequences[:, t + t1],
                    sequences[:, t + t2]
                ).astype(np.float32)

                squared_errors = (targets - outputs) ** 2
                total_squared_error += np.sum(squared_errors)

                predictions = (outputs > 0.5).astype(int)
                correct += np.sum(predictions == targets)
                total_predictions += batch_size

        if total_predictions == 0:
            return (0.0,)

        # MSE component (normalized)
        mse = total_squared_error / total_predictions
        mse_component = max(0.0, 1.0 - 4.0 * mse)

        # Accuracy component
        accuracy = correct / total_predictions

        fitness = 0.5 * mse_component + 0.5 * accuracy

    except Exception as e:
        print(f"Evaluation error: {e}")
        return (0.0,)

    return (fitness,)
import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from core.network import NetworkCompiler
from .metrics import classification_error, accuracy, mean_squared_error


class IrisClassificationFitness:
    def __init__(self):
        self.load_data()
        self.compiler = NetworkCompiler()

    def load_data(self):
        iris = load_iris()
        X, y = iris.data, iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        self.num_classes = 3
        self.input_size = 4

    def evaluate(self, chromosome):
        try:
            network = self.compiler.compile(chromosome)

            predictions = []
            for i in range(len(self.X_train)):
                input_data = self.X_train[i]
                network_output = network(input_data)

                if network_output.dim() == 0:
                    network_output = network_output.unsqueeze(0)

                if len(network_output) < self.num_classes:
                    padding = torch.zeros(self.num_classes - len(network_output))
                    network_output = torch.cat([network_output, padding])
                elif len(network_output) > self.num_classes:
                    network_output = network_output[:self.num_classes]

                predictions.append(network_output)

            predictions = torch.stack(predictions)
            predictions = torch.softmax(predictions, dim=1)

            error = classification_error(self.y_train, predictions)
            return error

        except Exception as e:
            return 1.0


class RegressionFitness:
    def __init__(self, target_function=None):
        self.target_function = target_function or self.quadratic_function
        self.generate_data()
        self.compiler = NetworkCompiler()

    def quadratic_function(self, x):
        return x ** 2

    def generate_data(self):
        X = np.linspace(-2, 2, 100).reshape(-1, 1)
        y = self.target_function(X.flatten())

        self.X_train = torch.tensor(X, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def evaluate(self, chromosome):
        try:
            network = self.compiler.compile(chromosome)

            predictions = []
            for i in range(len(self.X_train)):
                input_data = self.X_train[i]
                network_output = network(input_data)

                if network_output.dim() > 0:
                    network_output = network_output.mean()

                predictions.append(network_output)

            predictions = torch.stack(predictions)
            error = mean_squared_error(self.y_train, predictions)
            return error

        except Exception as e:
            return 1000.0


def get_fitness_function(task_type='iris_classification'):
    if task_type == 'iris_classification':
        return IrisClassificationFitness()
    elif task_type == 'regression':
        return RegressionFitness()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
from core.network import Network

# Cache for dataset
_dataset_cache = {}


def get_bracket_data(sequence_length=20, sequence_count=10000):
    cache_key = f'bracket_{sequence_length}_{sequence_count}'

    if cache_key not in _dataset_cache:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), 'data')
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, f'bracket_{sequence_length}_{sequence_count}.txt')

        if os.path.exists(filepath):
            sequences = []
            labels = []
            with open(filepath, 'r') as f:
                for line in f:
                    seq_str, label = line.strip().split(',')
                    seq = [{'x': 0, '(': 1, ')': 2, '[': 3, ']': 4, '{': 5, '}': 6}[c] for c in seq_str]
                    sequences.append(seq)
                    labels.append(int(label))
            X = np.array(sequences)
            y = np.array(labels)
        else:
            valid_sequences = []
            invalid_sequences = []
            char_map = {0: 'x', 1: '(', 2: ')', 3: '[', 4: ']', 5: '{', 6: '}'}

            while len(valid_sequences) < sequence_count // 2:
                seq = []
                stack = []
                for _ in range(sequence_length):
                    if random.random() < 0.15:
                        seq.append(0)
                    elif len(stack) > 0 and random.random() < 0.5:
                        bracket_pair = stack.pop()
                        seq.append(bracket_pair + 1)
                    else:
                        bracket_type = random.choice([1, 3, 5])
                        seq.append(bracket_type)
                        stack.append(bracket_type)

                if len(stack) == 0:
                    valid_sequences.append(''.join(char_map[c] for c in seq))

            while len(invalid_sequences) < sequence_count // 2:
                seq = [random.choice([0, 1, 2, 3, 4, 5, 6]) for _ in range(sequence_length)]
                stack = []
                valid = True
                for c in seq:
                    if c in [1, 3, 5]:
                        stack.append(c)
                    elif c in [2, 4, 6]:
                        if not stack or stack[-1] != c - 1:
                            valid = False
                            break
                        stack.pop()
                if not valid or len(stack) > 0:
                    invalid_sequences.append(''.join(char_map[c] for c in seq))

            with open(filepath, 'w') as f:
                for seq in valid_sequences:
                    f.write(f"{seq},1\n")
                for seq in invalid_sequences:
                    f.write(f"{seq},0\n")

            all_sequences = valid_sequences + invalid_sequences
            all_labels = [1] * len(valid_sequences) + [0] * len(invalid_sequences)

            X = np.array(
                [[{'x': 0, '(': 1, ')': 2, '[': 3, ']': 4, '{': 5, '}': 6}[c] for c in seq] for seq in all_sequences])
            y = np.array(all_labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y
        )

        _dataset_cache[cache_key] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    return _dataset_cache[cache_key]

if __name__ == '__main__':
    get_bracket_data(20, 2000)
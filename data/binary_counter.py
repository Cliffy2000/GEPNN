import numpy as np
import json


def generate_binary_counter(num_sequences=25, seq_length=10):
    data = []

    for _ in range(num_sequences):
        # Random binary sequence
        inputs = np.random.randint(0, 2, seq_length).tolist()

        outputs = []
        for t in range(seq_length):
            outputs.append(sum(inputs[:t+1]) % 2)

        data.append({
            'inputs': inputs,
            'outputs': outputs
        })

    # Save to file
    with open('binary_counter.json', 'w') as f:
        json.dump(data, f)


# Generate dataset
generate_binary_counter()
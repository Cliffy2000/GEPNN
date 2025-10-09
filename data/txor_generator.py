import numpy as np
import json


def generate_txor(k=2, num_sequences=1000, seq_length=20):
    data = []

    for _ in range(num_sequences):
        # Random binary sequence
        inputs = np.random.randint(0, 2, seq_length).tolist()

        # Compute XOR outputs
        outputs = []
        for t in range(seq_length):
            if t >= k:
                outputs.append(inputs[t] ^ inputs[t - k])
            else:
                outputs.append(0)  # undefined for first k steps

        data.append({
            'inputs': inputs,
            'outputs': outputs
        })

    # Save to file
    with open(f'txor_k{k}_n{num_sequences}_l{seq_length}.json', 'w') as f:
        json.dump({'k': k, 'sequences': data}, f)

    print(f"Generated {num_sequences} sequences, saved to txor_k{k}_n{num_sequences}_l{seq_length}.json")


# Generate dataset
generate_txor(k=1, num_sequences=25, seq_length=20)
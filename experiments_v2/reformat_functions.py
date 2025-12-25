import os
import json
import glob


def rename_functions(head_str):
    """Rename function names in head string."""
    # Rename unary first using placeholder to avoid collision
    # Then rename binary
    result = head_str

    # Step 1: Protect unary by renaming to placeholder
    result = result.replace("'sigmoid1'", "'__SIGMOID_UNARY__'")
    result = result.replace("'tanh1'", "'__TANH_UNARY__'")
    result = result.replace("'relu1'", "'__RELU_UNARY__'")

    # Step 2: Rename binary (no suffix) to '2' suffix
    result = result.replace("'sigmoid'", "'sigmoid2'")
    result = result.replace("'tanh'", "'tanh2'")
    result = result.replace("'relu'", "'relu2'")

    # Step 3: Rename placeholders to final unary names
    result = result.replace("'__SIGMOID_UNARY__'", "'sigmoid'")
    result = result.replace("'__TANH_UNARY__'", "'tanh'")
    result = result.replace("'__RELU_UNARY__'", "'relu'")

    return result


def update():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(script_dir, "..", "experiments_v2", "xor")

    for filepath in glob.glob(os.path.join(experiments_dir, "xor_*.json")):
        with open(filepath, 'r') as f:
            data = json.load(f)

        modified = False
        for result in data.get("results", []):
            expr = result.get("best_individual", {}).get("expression", {}).get("Individual", {})
            if "Head" in expr:
                old_head = expr["Head"]
                new_head = rename_functions(old_head)
                if old_head != new_head:
                    expr["Head"] = new_head
                    modified = True

        if modified:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Updated: {os.path.basename(filepath)}")
        else:
            print(f"No changes: {os.path.basename(filepath)}")


if __name__ == "__main__":
    update()
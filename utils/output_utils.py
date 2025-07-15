def print_separator(length=80):
    print("=" * length)


def print_section_break():
    print()
    print_separator()
    print()


def print_header(title, level=1):
    """Print a formatted header."""
    if level == 1:
        print()
        print_separator()
        print(f"{title.upper()}")
        print_separator()
    elif level == 2:
        print()
        print(f"{title}")
        print("-" * len(title))
    else:
        print(f"\n{title}:")


def print_test_info(inputs, weights, biases):
    """Print test parameters in a clean format."""
    print("\nTest Parameters:")
    print(f"  Inputs:  {format_dict(inputs)}")
    print(f"  Weights: {format_list(weights)}")
    print(f"  Biases:  {format_list(biases)}")


def format_dict(d):
    """Format dictionary for clean display."""
    items = [f"{k}={v}" for k, v in d.items()]
    return "{" + ", ".join(items) + "}"


def format_list(lst):
    """Format list with consistent decimal places."""
    if not lst:
        return "[]"
    if isinstance(lst[0], float):
        return "[" + ", ".join(f"{x:.1f}" for x in lst) + "]"
    return str(lst)


def print_calculation_step(step_num, description, calculation, result):
    """Print a calculation step in a clean format."""
    print(f"\n{step_num}. {description}")
    print(f"   {calculation}")
    print(f"   = {result}")


def print_result(expected, actual):
    """Print expected vs actual results."""
    print(f"\nExpected Output: {expected}")
    print(f"Actual Output:   {actual}")

    # Check if they match (within floating point tolerance)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if abs(expected - actual) < 0.001:
            print("✓ Test Passed")
        else:
            print("✗ Test Failed")


def print_tree_structure(description):
    """Print a tree structure description."""
    print("\nTree Structure:")
    print(f"  {description}")
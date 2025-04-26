#!/usr/bin/env python3
# filepath: /Users/waynechen/CS570100/hw1/verify_output.py

import sys


def read_patterns_file(filename):
    """Read patterns and their support values from a file."""
    patterns = {}
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("//"):  # Skip empty lines and comments
                continue

            # Split pattern and support
            parts = line.split(":")
            if len(parts) != 2:
                print(f"Warning: Ignoring malformed line: {line}")
                continue

            pattern, support = parts
            patterns[pattern] = float(support)

    return patterns


def compare_patterns(output_patterns, expected_patterns, tolerance=1e-4):
    """Compare output patterns with expected patterns."""
    # Check if all expected patterns are present
    missing_patterns = []
    for pattern in expected_patterns:
        if pattern not in output_patterns:
            missing_patterns.append(pattern)

    # Check if there are extra patterns in the output
    extra_patterns = []
    for pattern in output_patterns:
        if pattern not in expected_patterns:
            extra_patterns.append(pattern)

    # Check for support value discrepancies
    support_discrepancies = []
    for pattern in expected_patterns:
        if pattern in output_patterns:
            expected_support = expected_patterns[pattern]
            actual_support = output_patterns[pattern]
            if abs(expected_support - actual_support) > tolerance:
                support_discrepancies.append(
                    (pattern, expected_support, actual_support)
                )

    return missing_patterns, extra_patterns, support_discrepancies


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <your_output_file> <expected_output_file>")
        sys.exit(1)

    output_file = sys.argv[1]
    expected_file = sys.argv[2]

    # Read patterns from both files
    output_patterns = read_patterns_file(output_file)
    expected_patterns = read_patterns_file(expected_file)

    # Compare patterns
    missing, extra, support_discrepancies = compare_patterns(
        output_patterns, expected_patterns
    )

    # Print comparison results
    print(f"Your output: {len(output_patterns)} patterns")
    print(f"Expected output: {len(expected_patterns)} patterns")

    if not missing and not extra and not support_discrepancies:
        print("\n✅ SUCCESS: Your output matches the expected output!")
        return

    print("\n⚠️ DIFFERENCES FOUND:")

    if missing:
        print(f"\nMISSING PATTERNS ({len(missing)}):")
        for pattern in missing:
            print(f"  {pattern}: {expected_patterns[pattern]:.4f}")

    if extra:
        print(f"\nEXTRA PATTERNS ({len(extra)}):")
        for pattern in extra:
            print(f"  {pattern}: {output_patterns[pattern]:.4f}")

    if support_discrepancies:
        print(f"\nSUPPORT DISCREPANCIES ({len(support_discrepancies)}):")
        for pattern, expected, actual in support_discrepancies:
            print(
                f"  {pattern}: expected={expected:.4f}, actual={actual:.4f}, diff={abs(expected - actual):.6f}"
            )

    # Calculate overall match percentage
    total_expected = len(expected_patterns)
    correct_patterns = total_expected - len(missing) - len(support_discrepancies)
    match_percentage = (
        (correct_patterns / total_expected) * 100 if total_expected > 0 else 0
    )

    print(
        f"\nOverall match: {match_percentage:.2f}% ({correct_patterns}/{total_expected} patterns correct)"
    )


if __name__ == "__main__":
    main()

import json
import random
import math
from pathlib import Path
from collections import defaultdict

# Set a fixed seed to ensure the shuffle is always the same.
# This allows reproducing the exact same split in future runs.
random.seed(42)
path_dir = Path(__file__)

# File paths using relative references
# (Assumes the script and the input JSON are in the same directory)
input_file = 'ground_truth_gerado.json'
pool_file = 'pool.json'
test_file = 'ground_truth_test.json'

# Split ratios
pool_ratio = 0.8
test_ratio = 0.2

def split_dataset_by_difficulty(input_path, pool_path, test_path, pool_ratio, test_ratio):
    """
    Reads a JSON file of questions, splits it into two new files (pool and test)
    based on the provided ratios, and balances the difficulty categories.

    Args:
        input_path:  Path to the input JSON file containing all questions.
        pool_path:   Output path for the few-shot pool subset.
        test_path:   Output path for the evaluation test subset.
        pool_ratio:  Fraction of questions assigned to the pool (e.g., 0.8).
        test_ratio:  Fraction of questions assigned to the test set (e.g., 0.2).
    """
    # Error handling in case the file is not in the same directory
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
        print("Make sure it is in the same directory as this script or update 'input_file'.")
        return

    # Group questions by difficulty
    grouped_by_difficulty = defaultdict(list)
    for item in data:
        grouped_by_difficulty[item['dificuldade']].append(item)

    pool_data = []
    test_data = []

    # Process each difficulty category
    for difficulty, questions in grouped_by_difficulty.items():
        total_questions = len(questions)

        # Calculate the number of questions for the test set.
        # math.ceil ensures at least 1 question goes to test if the category is non-empty.
        num_test_questions = math.ceil(total_questions * test_ratio)

        # Ensure the number of test questions does not exceed the total available
        if num_test_questions > total_questions:
            num_test_questions = total_questions

        num_pool_questions = total_questions - num_test_questions

        # Shuffle questions to ensure a random selection
        random.shuffle(questions)

        # Split the questions
        test_set = questions[:num_test_questions]
        pool_set = questions[num_test_questions:]

        pool_data.extend(pool_set)
        test_data.extend(test_set)

        print(f"Difficulty: {difficulty}")
        print(f"  Total questions: {total_questions}")
        print(f"  Questions for '{pool_path}': {len(pool_set)} ({len(pool_set)/total_questions:.2%})")
        print(f"  Questions for '{test_path}': {len(test_set)} ({len(test_set)/total_questions:.2%})\n")

    # Save the new JSON files
    with open(pool_path, 'w', encoding='utf-8') as f:
        json.dump(pool_data, f, ensure_ascii=False, indent=4)

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    print(f"Process complete!")
    print(f"Total questions in '{pool_path}': {len(pool_data)}")
    print(f"Total questions in '{test_path}': {len(test_data)}")

# Run the algorithm
split_dataset_by_difficulty(path_dir.parent / input_file, pool_file, test_file, pool_ratio, test_ratio)

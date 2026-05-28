import json
import os
import csv
from pathlib import Path

# Short names for BBH tasks to create unique question_ids
BBH_SHORT_NAMES = {
    "date_understanding": "du",
    "disambiguation_qa": "dq",
    "penguins_in_a_table": "pit",
    "reasoning_about_colored_objects": "rco",
    "salient_translation_error_detection": "sted",
    "temporal_sequences": "ts",
    "tracking_shuffled_objects": "tso",
    "boolean_expressions": "be",
    "causal_judgement": "cj",
    "formal_fallacies": "ff",
    "navigate": "nav",
    "sports_understanding": "su",
    "web_of_lies": "wol",
    "dyck_languages": "dl",
    "multistep_arithmetic_two": "mat",
    "object_counting": "oc",
    "word_sorting": "ws",
}


def update_bbh_question_ids(path: str):
    path = Path(path)
    if not path.exists():
        print(f"Path {path} does not exist.")
        return

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.jsonl', '.csv')):
                filepath = Path(root) / file
                task_name = Path(root).name  # immediate parent directory name
                if task_name in BBH_SHORT_NAMES:
                    prefix = BBH_SHORT_NAMES[task_name]
                    print(f"Updating {filepath} with prefix {prefix}")
                    if file.endswith('.jsonl'):
                        updated_lines = []
                        with open(filepath, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    data = json.loads(line)
                                    if 'question_id' in data:
                                        original_id = data['question_id']
                                        data['question_id'] = f"{prefix}_{original_id}"
                                    updated_lines.append(json.dumps(data))
                        # Write back
                        with open(filepath, 'w') as f:
                            for line in updated_lines:
                                f.write(line + '\n')
                    elif file.endswith('.csv'):
                        with open(filepath, 'r', newline='') as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                        for row in rows:
                            if 'question_id' in row:
                                row['question_id'] = f"{prefix}_{row['question_id']}"
                        with open(filepath, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python update_bbh_question_ids.py <path>")
        sys.exit(1)
    update_bbh_question_ids(sys.argv[1])
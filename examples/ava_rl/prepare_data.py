"""Map supplied Ava JSONL rows to slime prompts without changing their tests."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = {}
    # Preserve dataset rows and input ordering; slime owns shuffling and sampling.
    with args.output.open("x") as output:
        for source, path in [("ava_math", args.math), ("ava_terminal", args.terminal)]:
            count = 0
            with path.open() as stream:
                for source_index, line in enumerate(stream):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if source == "ava_math":
                        prompt, label = row["problem"], row["expected_answer"]
                        ava_row = {"prompt": prompt, "label": label}
                    else:
                        prompt, label = row["prompt"], None
                        ava_row = row
                    output.write(
                        json.dumps(
                            {
                                "prompt": [{"role": "user", "content": prompt}],
                                "label": label,
                                "metadata": {"source": source, "source_index": source_index, "ava_row": ava_row},
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                    count += 1
            counts[source] = count
    print(json.dumps(counts))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os

outputs_path = "length_norm_outputs"
for alpha_suffix in ["0", "0.25", "0.5", "0.75", "1"]:
    with open(os.path.join(outputs_path, f"output{alpha_suffix}.en"), "r") as f:
        tot_length = 0
        num_lines = 0
        for line in f:
            length = len(line.rstrip("\n"))
            tot_length += length
            num_lines += 1
        avg_length = tot_length / num_lines
        print(f"alpha={alpha_suffix}: avg_length={avg_length}")

#!/usr/bin/env python3
import os

outputs_path = "stopping_criteria_outputs"
for file in os.listdir(outputs_path):
    if file.startswith("output"):
        with open(os.path.join(outputs_path, file), "r") as f:
            tot_length = 0
            num_lines = 0
            for line in f:
                length = len(line.rstrip("\n"))
                tot_length += length
                num_lines += 1
            avg_length = tot_length / num_lines
            print(f"{file}: avg_length={avg_length}")

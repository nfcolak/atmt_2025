#!/usr/bin/env python3
import os
import re

outputs_path = "stopping_criteria_outputs"
with open(os.path.join(outputs_path, "stopping_criteria.out"), "r") as f:
    full_txt = f.read()
    settings = re.findall("Translation with (.+)\n", full_txt)
    bleu_scores = re.findall("\nBLEU score: ([\d\.]+)\n", full_txt)
    times = re.findall("([\d\.]+) seconds\n", full_txt)
    for setting, bleu, time in zip(settings, bleu_scores, times):
        print(f"{setting}\tBLEU: {bleu}\tTime: {time}s")

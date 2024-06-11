#!/bin/bash

pip install imblearn

# List of languages
# "am" "su" "sw" "ka" "ne" "ug" "yo" "ur" "mk" "mr" "bn" "te" "uz" "az" "sl" "lv" "ro" "he" "cy" "bg" "sk" "da"
languages=("si")

# PPMI emb space
ppmi_space="single"

# Iterate through each language and run the Python script
for language in "${languages[@]}"; do
    echo "Running retrofit.py for language: $language with kernel: $kernel"
    python embs_stats.py --language "$language" --ppmi_type "$ppmi_space"
done
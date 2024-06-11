#!/bin/bash

pip install imblearn

# List of languages
# all_languages=("si" "am" "su" "sw" "ka" "ne" "ug" "yo" "ur" "mk" "mr" "bn" "te" "uz" "az" "sl" "lv" "ro" "he" "cy" "bg" "sk" "da")
all_languages=("ug" "mk")

# Kernel type
kernel="rbf"


# Iterate through each language and run the Python script
for language in "${all_languages[@]}"; do
    echo "Running retrofit.py for language: $language with kernel: $kernel"
    python xlmr_sa.py --language "$language" --kernel "$kernel"
done
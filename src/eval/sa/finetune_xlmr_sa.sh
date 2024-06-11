#!/bin/bash

pip install huggingface_hub

# List of languages 
all_languages=("si" "am" "su" "sw" "ka" "ne" "ur"  "ug" "yo" "mk" "mr" "bn" "te" "uz" "az" "sl" "lv" "ro" "he" "cy" "bg" "sk" "da")


# Iterate through each language and run the Python script
for language in "${all_languages[@]}"; do
    echo "Running for language: $language"
    python finetune_xlmr_sa.py --language "$language" --huggingface_token "-" --repo_name xlm-r
done
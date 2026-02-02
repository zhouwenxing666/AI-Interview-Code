import os
from datasets import load_dataset

# Define target directory
target_dir = "/Users/zhouwenxing/Documents/Projects/Python/AI-Interview-Code/learning/stanford-cs336/assignment1/data/tinystories"
os.makedirs(target_dir, exist_ok=True)

# Download validation split from huggingface
print("Downloading TinyStories validation split...")
try:
    # Use stream=True to avoid downloading the whole thing if we just want a peek or if it allows saving iterator?
    # But to_json needs full dataset or writes line by line.
    # TinyStories validation is small enough.
    ds = load_dataset("roneneldan/TinyStories", split="validation")
    output_path = os.path.join(target_dir, "validation.json")
    ds.to_json(output_path)
    print(f"Saved validation dataset to {output_path}")
    
    # Also check if we need train split? The notebook might use it later.
    # The traceback showed the user trying to load 'validation'.
    # If they need 'train', they might hit another error later, but let's fix the current one first.
    
except Exception as e:
    print(f"Error downloading dataset: {e}")

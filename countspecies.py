import os
import numpy as np

base_path = "output-dataset"
species_counts = {}
unreadable_files = []

print("Checking base path:", base_path)
if not os.path.exists(base_path):
    print("ERROR: Base path does not exist.")
    exit()
for species in sorted(os.listdir(base_path)):
    species_path = os.path.join(base_path, species)
    if os.path.isdir(species_path):
        npy_files = [f for f in os.listdir(species_path) if f.endswith(".npy")]
        count = 0
        for f in npy_files:
            file_path = os.path.join(species_path, f)
            try:
                _ = np.load(file_path)
                count += 1
            except Exception as e:
                unreadable_files.append((file_path, str(e)))
        species_counts[species] = count
        print(f"Species: {species}, .npy files counted: {count}")

empty = [s for s, count in species_counts.items() if count == 0]
few_samples = [s for s, count in species_counts.items() if 0 < count <= 2]

print(f"\nTotal species: {len(species_counts)}")
print(f"Species with 0 readable .npy files: {len(empty)}")
for s in empty:
    print(f"- {s}")

print(f"\nSpecies with fewer than 2 samples (need removal for train_test_split): {len(few_samples)}")
for s in few_samples:
    print(f"- {s}")

if unreadable_files:
    print("\nUnreadable .npy files:")
    for file_path, error_msg in unreadable_files:
        print(f"- {file_path}: {error_msg}")
else:
    print("\nAll files loaded successfully.")

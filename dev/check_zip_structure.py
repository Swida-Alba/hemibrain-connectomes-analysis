import zipfile
import os

zip_path = "/Users/apple/Documents/GitHub/hemibrain-connectomes-analysis-v3.1/datasets/flywire_FAFB_v783/downloads/sk_lod1_783_healed.zip"

if not os.path.exists(zip_path):
    print(f"Zip file not found: {zip_path}")
else:
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            print(f"Total files in zip: {len(z.namelist())}")
            print("First 10 files:")
            for f in z.namelist()[:10]:
                print(f)
    except Exception as e:
        print(f"Error reading zip: {e}")

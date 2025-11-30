import os
import zipfile
import numpy as np
import pandas as pd
import glob

def inspect_skeletons():
    # Find the zip file
    zip_paths = glob.glob("datasets/**/*skeletons.zip", recursive=True) + \
                glob.glob("datasets/**/sk_lod1_783_healed.zip", recursive=True) + \
                glob.glob("datasets/**/downloads/sk_lod1_783_healed.zip", recursive=True)
    
    if not zip_paths:
        print("No skeleton zip found.")
        return

    zip_path = zip_paths[0]
    print(f"Inspecting {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as z:
        swc_files = [f for f in z.namelist() if f.endswith('.swc')]
        if not swc_files:
            print("No SWC files in zip.")
            return
        
        print(f"Found {len(swc_files)} SWC files. Inspecting first 5...")
        
        all_x = []
        all_y = []
        all_z = []
        types = set()

        for i, filename in enumerate(swc_files[:5]):
            with z.open(filename) as f:
                content = f.read().decode('utf-8')
                lines = [l.strip() for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
                
                # Print first line of first file to show format
                if i == 0:
                    print(f"Sample line from {filename}: {lines[0]}")
                
                for line in lines:
                    parts = line.split()
                    # SWC: id type x y z radius parent
                    if len(parts) >= 7:
                        types.add(parts[1])
                        all_x.append(float(parts[2]))
                        all_y.append(float(parts[3]))
                        all_z.append(float(parts[4]))

        # Check types
        print(f"\nUnique types found: {types}")
        try:
            types_int = [int(t) for t in types]
            print("Types can be converted to int.")
        except ValueError:
            print("Types contain non-integers.")

        # Check coords
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        min_z, max_z = min(all_z), max(all_z)
        
        print(f"\nCoordinate Ranges:")
        print(f"X: {min_x} to {max_x}")
        print(f"Y: {min_y} to {max_y}")
        print(f"Z: {min_z} to {max_z}")
        
        max_val = max(abs(max_x), abs(max_y), abs(max_z), abs(min_x), abs(min_y), abs(min_z))
        print(f"Max absolute coordinate value: {max_val}")
        
        if max_val < 65500:
            print("Fits in float16 (max ~65504).")
        else:
            print("Does NOT fit in float16.")

        # Check for decimal precision
        print("\nChecking precision...")
        all_coords = all_x + all_y + all_z
        non_integers = [c for c in all_coords if c != int(c)]
        if not non_integers:
            print("All coordinates are integers.")
        else:
            print(f"Found {len(non_integers)} non-integer coordinates out of {len(all_coords)}.")
            print(f"Sample non-integers: {non_integers[:5]}")
            
            # Check max decimal places
            max_decimals = 0
            for c in non_integers[:1000]: # Check first 1000
                s = str(c)
                if '.' in s:
                    decimals = len(s.split('.')[1])
                    max_decimals = max(max_decimals, decimals)
            print(f"Max decimal places observed (sample): {max_decimals}")

        # Check radius precision
        print("\nChecking radius precision...")
        all_radii = []
        with zipfile.ZipFile(zip_path, 'r') as z:
            for filename in swc_files[:5]:
                with z.open(filename) as f:
                    content = f.read().decode('utf-8')
                    lines = [l.strip() for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 7:
                            all_radii.append(float(parts[5]))
        
        non_int_radii = [r for r in all_radii if r != int(r)]
        if not non_int_radii:
            print("All radii are integers.")
        else:
            print(f"Found {len(non_int_radii)} non-integer radii out of {len(all_radii)}.")
            print(f"Sample non-integer radii: {non_int_radii[:5]}")

if __name__ == "__main__":
    inspect_skeletons()

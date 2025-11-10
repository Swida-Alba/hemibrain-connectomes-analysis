#!/usr/bin/env python3
"""
Complete Project Reorganization Script
Reorganizes the entire project structure and fixes all imports
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()

def backup_current_state():
    """Create a backup of current state"""
    print("Creating backup of current state...")
    backup_dir = BASE_DIR / '.backup_before_reorganization'
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir()
    
    # Backup critical files
    for pattern in ['*.py', '*.ipynb', '*.html']:
        for file in BASE_DIR.glob(pattern):
            if file.name not in ['setup.py', 'reorganize_project.py', 'complete_reorganization.py']:
                shutil.copy2(file, backup_dir / file.name)
    print(f"  ✓ Backup created at: {backup_dir}")

def clean_existing_structure():
    """Clean up any partial reorganization"""
    print("\nCleaning existing structure...")
    
    # Remove duplicate files in src/
    src_duplicates = ['datasets', 'neuprint_cache']
    for item in src_duplicates:
        path = BASE_DIR / 'src' / item
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  ✓ Removed duplicate: src/{item}")

def move_files_to_structure():
    """Move files to their proper locations"""
    print("\nMoving files to proper structure...")
    
    moves = {
        # Core modules to src/
        'coana.py': 'src/coana.py',
        'statvis.py': 'src/statvis.py',
        'vispath.py': 'src/vispath.py',
        'ManageCache.py': 'src/core/cache_manager.py',
        
        # Scripts to scripts/
        'FindDirect.py': 'scripts/FindDirect.py',
        'FindPath.py': 'scripts/FindPath.py',
        'FindPath_Kun.py': 'scripts/FindPath_Kun.py',
        'FindPath_VTaMe.py': 'scripts/FindPath_VTaMe.py',
        'FindPath_PPL1_VTaMe.py': 'scripts/FindPath_PPL1_VTaMe.py',
        'PlotPath.py': 'scripts/PlotPath.py',
        'update_heatmap_html.py': 'scripts/update_heatmap_html.py',
    }
    
    moved = 0
    for src, dst in moves.items():
        src_path = BASE_DIR / src
        dst_path = BASE_DIR / dst
        
        if not src_path.exists():
            # Check if already moved
            if dst_path.exists():
                print(f"  ✓ Already in place: {dst}")
                continue
            else:
                print(f"  ⚠ Not found: {src}")
                continue
        
        # Remove destination if it exists
        if dst_path.exists():
            dst_path.unlink()
        
        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(str(src_path), str(dst_path))
        print(f"  ✓ Moved: {src} → {dst}")
        moved += 1
    
    print(f"\nMoved {moved} files")

def update_script_imports():
    """Update import statements in all scripts"""
    print("\nUpdating import statements in scripts...")
    
    scripts_dir = BASE_DIR / 'scripts'
    if not scripts_dir.exists():
        print("  ⚠ Scripts directory not found")
        return
    
    import_header = '''import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

'''
    
    updated = 0
    for script in scripts_dir.glob('*.py'):
        try:
            content = script.read_text(encoding='utf-8')
            
            # Check if it imports from coana, statvis, or vispath
            if any(module in content for module in ['from coana import', 'import coana', 
                                                      'from statvis import', 'import statvis',
                                                      'from vispath import', 'import vispath']):
                
                # Check if already has the path setup
                if 'sys.path.insert(0, str(Path(__file__).parent.parent' not in content:
                    # Find where to insert (after initial imports, before main imports)
                    lines = content.split('\n')
                    
                    # Find first import line
                    first_import_idx = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith('import ') or line.strip().startswith('from '):
                            first_import_idx = i
                            break
                    
                    # Insert path setup before first import
                    lines.insert(first_import_idx, import_header.rstrip())
                    
                    # Write back
                    script.write_text('\n'.join(lines), encoding='utf-8')
                    print(f"  ✓ Updated: {script.name}")
                    updated += 1
                else:
                    print(f"  ○ Already updated: {script.name}")
        except Exception as e:
            print(f"  ✗ Error updating {script.name}: {e}")
    
    print(f"\nUpdated {updated} scripts")

def create_init_files():
    """Create __init__.py files for packages"""
    print("\nCreating __init__.py files...")
    
    packages = [
        'src',
        'src/core',
        'src/plotting',
        'src/utils',
        'tests',
    ]
    
    for pkg in packages:
        pkg_path = BASE_DIR / pkg
        pkg_path.mkdir(parents=True, exist_ok=True)
        init_file = pkg_path / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print(f"  ✓ Created: {pkg}/__init__.py")

def update_cache_directory():
    """Ensure cache directory is properly named"""
    print("\nUpdating cache directory...")
    
    old_cache = BASE_DIR / 'neuprint_cache'
    new_cache = BASE_DIR / 'cache'
    
    # Check if old cache exists in root
    if old_cache.exists() and not new_cache.exists():
        shutil.move(str(old_cache), str(new_cache))
        print(f"  ✓ Renamed: neuprint_cache → cache")
    elif old_cache.exists() and new_cache.exists():
        print(f"  ⚠ Both cache directories exist, keeping 'cache'")
        shutil.rmtree(old_cache)
    elif new_cache.exists():
        print(f"  ✓ Cache directory already correct")
    else:
        new_cache.mkdir(exist_ok=True)
        print(f"  ✓ Created: cache/")

def create_directory_structure():
    """Ensure all necessary directories exist"""
    print("\nCreating directory structure...")
    
    dirs = [
        'src',
        'src/core',
        'src/plotting',
        'src/utils',
        'scripts',
        'notebooks',
        'tests',
        'test_output',
        'test_output/html',
        'test_output/data',
        'output',
        'output/sankey',
        'output/networks',
        'output/plots',
        'cache',
    ]
    
    for d in dirs:
        path = BASE_DIR / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Ensured: {d}/")

def update_gitignore():
    """Update .gitignore with proper entries"""
    print("\nUpdating .gitignore...")
    
    gitignore_path = BASE_DIR / '.gitignore'
    
    new_entries = [
        '',
        '# Reorganization updates',
        'output/',
        'cache/',
        'test_output/',
        '*.pyc',
        '__pycache__/',
        '.backup_before_reorganization/',
    ]
    
    if gitignore_path.exists():
        existing = gitignore_path.read_text()
        # Only add if not already present
        if '# Reorganization updates' not in existing:
            with open(gitignore_path, 'a') as f:
                f.write('\n'.join(new_entries))
            print(f"  ✓ Updated .gitignore")
        else:
            print(f"  ○ .gitignore already updated")
    else:
        gitignore_path.write_text('\n'.join(new_entries))
        print(f"  ✓ Created .gitignore")

def verify_structure():
    """Verify the reorganized structure"""
    print("\n" + "="*70)
    print("VERIFYING REORGANIZED STRUCTURE")
    print("="*70)
    
    checks = {
        'src/coana.py': 'Core module: coana.py',
        'src/statvis.py': 'Core module: statvis.py',
        'src/vispath.py': 'Core module: vispath.py',
        'src/core/cache_manager.py': 'Cache manager',
        'scripts/FindPath.py': 'Main script: FindPath.py',
        'scripts/FindDirect.py': 'Main script: FindDirect.py',
        'cache': 'Cache directory',
    }
    
    all_ok = True
    for path, desc in checks.items():
        full_path = BASE_DIR / path
        if full_path.exists():
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ Missing: {desc}")
            all_ok = False
    
    print("="*70)
    if all_ok:
        print("✅ ALL CHECKS PASSED")
    else:
        print("⚠️  SOME CHECKS FAILED - Please review")
    print("="*70)

def main():
    """Main reorganization function"""
    print("="*70)
    print("COMPLETE PROJECT REORGANIZATION")
    print("="*70)
    print()
    
    response = input("This will reorganize the entire project. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Reorganization cancelled.")
        return
    
    print()
    
    # Step 1: Backup
    backup_current_state()
    
    # Step 2: Create structure
    create_directory_structure()
    
    # Step 3: Clean up
    clean_existing_structure()
    
    # Step 4: Move files
    move_files_to_structure()
    
    # Step 5: Update imports
    update_script_imports()
    
    # Step 6: Create __init__ files
    create_init_files()
    
    # Step 7: Update cache
    update_cache_directory()
    
    # Step 8: Update gitignore
    update_gitignore()
    
    # Step 9: Verify
    verify_structure()
    
    print("\n" + "="*70)
    print("✅ REORGANIZATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Test scripts: python scripts/FindPath.py")
    print("2. Review documentation in docs/")
    print("3. Commit changes to git")
    print("\nBackup available at: .backup_before_reorganization/")
    print("="*70)

if __name__ == '__main__':
    main()

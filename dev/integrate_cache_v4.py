"""
Script to integrate v4.0 cache methods into coana.py
Replaces old hash-based cache methods with unified database approach.
"""

import re

# Read cache_v4_methods.py to get the new methods
with open('cache_v4_methods.py', 'r') as f:
    v4_methods = f.read()

# Extract just the method definitions (skip the class wrapper and imports)
# We want everything after "# ============================================================================"
v4_methods_start = v4_methods.find('    # ============================================================================')
v4_methods_content = v4_methods[v4_methods_start:]

# Remove the class indentation (4 spaces)
v4_methods_content = '\n'.join([line[4:] if line.startswith('    ') else line 
                                 for line in v4_methods_content.split('\n')])

# Read coana.py
with open('coana.py', 'r') as f:
    coana_content = f.read()

# Find the section to replace
# Start: after _ensure_complete_dataset method
# End: before _fetch_connections_with_cache (but we're replacing this too!)

# Find start marker
start_marker = "    def _get_neuron_registry_path(self):"
start_index = coana_content.find(start_marker)

if start_index == -1:
    print("❌ Could not find start marker in coana.py")
    exit(1)

# Find end marker - we want to replace everything up to (but not including) the next major method
# after all the cache methods. Let's find "def InitializeNeuronInfo"
end_marker = "    def InitializeNeuronInfo(self):"
end_index = coana_content.find(end_marker, start_index)

if end_index == -1:
    print("❌ Could not find end marker in coana.py")
    exit(1)

# Build new content
new_content = (
    coana_content[:start_index] +
    "    " + v4_methods_content.replace('\n', '\n    ') + "\n\n" +  # Add indentation
    coana_content[end_index:]
)

# Write back
with open('coana.py', 'w') as f:
    f.write(new_content)

print("✅ Successfully integrated v4.0 cache methods into coana.py")
print(f"   Replaced {end_index - start_index} characters")
print(f"   New code is {len(v4_methods_content)} characters")

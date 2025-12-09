import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from comparison import LabelMapper

target_map = LabelMapper(
  target_mapping_dict={
    'flywire_FAFB_v783': [[720575940634984800,720575940627933336,720575940625254636,720575940619074049]],
    'flywire_BANC_v626': [[720575941671706023,720575941645496264,720575941568371246]],
    'male-cns:v0.9': [[11901,14633,12254,13531]],
  },
  target_labels=['E cells'],
)

dataset = 'flywire_FAFB_v783'
body_id = '720575940634984800'

print(f"Testing mapping for {body_id} in {dataset}")
mapped = target_map.get_std_label(dataset, body_id, 'target')
print(f"Mapped label: '{mapped}'")

expected = 'E cells'
if mapped == expected:
    print("SUCCESS: Mapping works as expected.")
else:
    print(f"FAILURE: Expected '{expected}', got '{mapped}'")

#!/usr/bin/env python3
"""Test hemisphere suffix application directly"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/apple/Documents/GitHub/hemibrain-connectomes-analysis-v3.1/src')

# Test _apply_hemisphere_suffix_to_conn_df directly with mock data
class MockFNC:
    def __init__(self):
        self.separate_hemispheres = True
    
    def _normalize_hemisphere_value(self, val):
        if pd.isna(val) or val is None:
            return 'U'
        val_str = str(val).strip().upper()
        if val_str in ['L', 'LEFT', '0.0', '0']:
            return 'L'
        if val_str in ['R', 'RIGHT', '1.0', '1']:
            return 'R'
        return 'U'
    
    def _append_hemi_suffix(self, type_val, hemi_code):
        if type_val is None or (isinstance(type_val, float) and pd.isna(type_val)):
            return type_val
        type_str = str(type_val)
        if type_str.endswith('_L') or type_str.endswith('_R') or type_str.endswith('_U'):
            return type_str
        return f"{type_str}_{hemi_code}"
    
    def _apply_hemisphere_suffix_to_conn_df(self, conn_df):
        if conn_df is None or conn_df.empty or not self.separate_hemispheres:
            return conn_df
        def _get_hemi_code(row, side: str) -> str:
            code_col = f"hemisphere_code_{side}"
            if code_col in row and pd.notna(row[code_col]):
                return str(row[code_col])
            hemi_col = f"hemisphere_{side}"
            if hemi_col in row and pd.notna(row[hemi_col]):
                return self._normalize_hemisphere_value(row[hemi_col])
            inst_col = f"instance_{side}"
            if inst_col in row and isinstance(row[inst_col], str):
                if row[inst_col].endswith('_R'):
                    return 'R'
                if row[inst_col].endswith('_L'):
                    return 'L'
            return 'U'

        conn_df = conn_df.copy()
        if 'type_pre' in conn_df.columns:
            conn_df['type_pre'] = conn_df.apply(lambda row: self._append_hemi_suffix(row['type_pre'], _get_hemi_code(row, 'pre')), axis=1)
        if 'type_post' in conn_df.columns:
            conn_df['type_post'] = conn_df.apply(lambda row: self._append_hemi_suffix(row['type_post'], _get_hemi_code(row, 'post')), axis=1)
        return conn_df

# Create test data
test_data = {
    'bodyId_pre': ['11454', '11454', '16674', '16674'],
    'bodyId_post': ['20046', '19513', '20046', '19513'],
    'weight': [100, 50, 80, 60],
    'type_pre': ['MeVPaMe1', 'MeVPaMe1', 'aMe10', 'aMe10'],
    'type_post': ['MeVPLo2', 'aMe10', 'MeVPLo2', 'aMe10'],
    'instance_pre': ['MeVPaMe1_L', 'MeVPaMe1_L', 'aMe10_L', 'aMe10_L'],
    'instance_post': ['MeVPLo2_L', 'aMe10_R', 'MeVPLo2_L', 'aMe10_R'],
}
conn_df = pd.DataFrame(test_data)

print("=== BEFORE applying hemisphere suffix ===")
print(conn_df[['type_pre', 'type_post', 'instance_pre', 'instance_post']].to_string())

fnc = MockFNC()
result = fnc._apply_hemisphere_suffix_to_conn_df(conn_df)

print("\n=== AFTER applying hemisphere suffix ===")
print(result[['type_pre', 'type_post', 'instance_pre', 'instance_post']].to_string())

# Check
print("\n=== Verification ===")
print(f"type_pre has _L suffix: {result['type_pre'].str.endswith('_L').any()}")
print(f"type_pre has _R suffix: {result['type_pre'].str.endswith('_R').any()}")
print(f"type_post has _L suffix: {result['type_post'].str.endswith('_L').any()}")
print(f"type_post has _R suffix: {result['type_post'].str.endswith('_R').any()}")

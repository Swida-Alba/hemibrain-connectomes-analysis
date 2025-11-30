"""
LabelMapper - Cross-dataset label standardization for comparison analysis.

This module provides the LabelMapper class for standardizing neuron identifiers
across different datasets with varying naming conventions.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Union, Optional, Any
from collections import defaultdict


class LabelMapper:
    """
    Cross-dataset label standardization for comparison analysis.
    
    Handles mapping logic across ALL datasets (not embedded in DatasetConfig).
    Supports CSV and JSON input formats, with asymmetric neuron counts per dataset.
    
    Key Features:
    - Centralized mapping logic for cross-dataset consistency
    - Supports CSV format (one file for sources, one for targets, ALL datasets)
    - Supports JSON format with nested group structure
    - Auto-generates missing labels using {original_id}_etc format
    - Exports mapping to parameters.json for reproducibility
    - Provides mapping summary for user verification
    
    CSV Format Example:
        std_label,hemibrain_v1_2_1,male-cns_v0_9,flywire_FAFB_v783
        aMe12_grp1,aMe12,aMe12,720575940610453042
        aMe12_grp1,aMe12_R,aMe12-like,720575940610453043
        aMe12_grp1,,aMe12_variant,
        
    JSON Format Example:
        {
            "source_mapping": {
                "std_label": ["aMe12_grp1", "aMe12_grp2"],
                "hemibrain:v1.2.1": [["aMe12", "aMe12_R"], ["aMe12_L"]]
            }
        }
    
    Example Usage:
        >>> # From files
        >>> mapper = LabelMapper(
        ...     source_mapping_file='mappings/source_mapping.csv',
        ...     target_mapping_file='mappings/target_mapping.csv'
        ... )
        
        >>> # From dictionaries
        >>> mapper = LabelMapper(
        ...     source_mapping_dict={
        ...         'hemibrain:v1.2.1': [['aMe12', 'aMe12_R']],
        ...         'male-cns:v0.9': [['aMe12', 'aMe12-like']]
        ...     },
        ...     source_labels=['aMe12_grp']
        ... )
        
        >>> # Get standardized label
        >>> std = mapper.get_std_label('hemibrain:v1.2.1', 'aMe12', 'source')
    """
    
    def __init__(
        self,
        source_mapping_file: Optional[str] = None,
        target_mapping_file: Optional[str] = None,
        source_mapping_dict: Optional[Dict] = None,
        target_mapping_dict: Optional[Dict] = None,
        source_labels: Optional[List[str]] = None,
        target_labels: Optional[List[str]] = None,
    ):
        """
        Initialize LabelMapper.
        
        Args:
            source_mapping_file: Path to CSV or JSON file for source mappings
            target_mapping_file: Path to CSV or JSON file for target mappings
            source_mapping_dict: Dictionary mapping datasets to source neurons
            target_mapping_dict: Dictionary mapping datasets to target neurons
            source_labels: Standard labels for source groups (used with dict input)
            target_labels: Standard labels for target groups (used with dict input)
        """
        # Internal storage: {std_label: {dataset: [neuron_ids]}}
        self._source_mapping: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        self._target_mapping: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        
        # Reverse lookup: {dataset: {neuron_id: std_label}}
        self._source_reverse: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._target_reverse: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Store file paths for export
        self._source_file = source_mapping_file
        self._target_file = target_mapping_file
        
        # Load from files if provided
        if source_mapping_file:
            self._load_source_from_file(source_mapping_file)
        elif source_mapping_dict:
            self._load_source_from_dict(source_mapping_dict, source_labels or [])
        
        if target_mapping_file:
            self._load_target_from_file(target_mapping_file)
        elif target_mapping_dict:
            self._load_target_from_dict(target_mapping_dict, target_labels or [])
        
        # Build reverse lookups
        self._build_reverse_lookups()
    
    def _load_source_from_file(self, filepath: str) -> None:
        """Load source mapping from CSV or JSON file."""
        if filepath.endswith('.json'):
            self._load_from_json(filepath, 'source')
        else:
            self._load_from_csv(filepath, 'source')
    
    def _load_target_from_file(self, filepath: str) -> None:
        """Load target mapping from CSV or JSON file."""
        if filepath.endswith('.json'):
            self._load_from_json(filepath, 'target')
        else:
            self._load_from_csv(filepath, 'target')
    
    def _load_from_csv(self, filepath: str, role: str) -> None:
        """
        Load mapping from CSV file.
        
        Supports two CSV formats:
        
        Format 1 (New - Expanded): Each row is one pattern, grouped by custom_label
            custom_label,std_pattern,dataset1,dataset2,notes
            MBON14,MBON14.*_R,MBON14.*_R,MBON14.*_R,Right hemisphere
            MBON14,MBON14.*_L,MBON14.*_L,MBON14.*_L,Left hemisphere
            
        Format 2 (Legacy): std_label with semicolon-separated patterns
            std_label,dataset1,dataset2,dataset3
            label1,id1;id2,id3,id4
        
        Args:
            filepath: Path to CSV file
            role: 'source' or 'target'
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Mapping file not found: {filepath}")
        
        df = pd.read_csv(filepath, dtype=str)
        
        # Detect format by checking for 'custom_label' column (new format)
        # or 'std_label' column (legacy format)
        if 'custom_label' in df.columns:
            self._load_csv_expanded_format(df, role)
        elif 'std_label' in df.columns:
            self._load_csv_legacy_format(df, role)
        else:
            raise ValueError(f"CSV must have 'custom_label' or 'std_label' column. Found: {df.columns.tolist()}")
    
    def _load_csv_expanded_format(self, df: pd.DataFrame, role: str) -> None:
        """
        Load from new expanded CSV format where each row is one pattern.
        
        Format:
            custom_label,std_pattern,dataset1,dataset2,notes
            MBON14,MBON14.*_R,MBON14.*_R,MBON14.*_R,Right hemisphere
            MBON14,MBON14.*_L,MBON14.*_L,MBON14.*_L,Left hemisphere
        
        Rows with the same custom_label are grouped together.
        
        Args:
            df: DataFrame with custom_label column
            role: 'source' or 'target'
        """
        # Identify dataset columns (exclude custom_label, std_pattern, notes, description, etc.)
        exclude_cols = {'custom_label', 'std_pattern', 'notes', 'description', 'comment'}
        dataset_cols = [c for c in df.columns if c.lower() not in exclude_cols]
        
        mapping = self._source_mapping if role == 'source' else self._target_mapping
        
        for _, row in df.iterrows():
            custom_label = row.get('custom_label', '')
            
            # Auto-generate label if missing
            if pd.isna(custom_label) or str(custom_label).strip() == '':
                # Use std_pattern or first non-empty value
                if 'std_pattern' in row and pd.notna(row['std_pattern']):
                    custom_label = self.auto_generate_label(str(row['std_pattern']))
                else:
                    for col in dataset_cols:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            custom_label = self.auto_generate_label(str(row[col]))
                            break
                if pd.isna(custom_label) or str(custom_label).strip() == '':
                    continue  # Skip rows with no valid data
            
            custom_label = str(custom_label).strip()
            
            for col in dataset_cols:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    # Convert column name back to dataset format
                    dataset = self._unsanitize_dataset_name(col)
                    neuron_id = str(value).strip()
                    
                    # Try to convert to int if it looks like a bodyId
                    try:
                        if neuron_id.isdigit():
                            neuron_id = int(neuron_id)
                    except:
                        pass
                    
                    # Add to mapping (rows with same custom_label are aggregated)
                    if neuron_id not in mapping[custom_label][dataset]:
                        mapping[custom_label][dataset].append(neuron_id)
    
    def _load_csv_legacy_format(self, df: pd.DataFrame, role: str) -> None:
        """
        Load from legacy CSV format with semicolon-separated patterns.
        
        Format:
            std_label,dataset1,dataset2,dataset3
            label1,id1;id2,id3,id4
        
        Args:
            df: DataFrame with std_label column
            role: 'source' or 'target'
        """
        # Get dataset columns (all except std_label)
        dataset_cols = [c for c in df.columns if c != 'std_label']
        
        mapping = self._source_mapping if role == 'source' else self._target_mapping
        
        for _, row in df.iterrows():
            std_label = row['std_label']
            
            # Auto-generate label if missing
            if pd.isna(std_label) or str(std_label).strip() == '':
                # Find first non-empty value in row for auto-generation
                for col in dataset_cols:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        std_label = self.auto_generate_label(str(row[col]).split(';')[0])
                        break
                if pd.isna(std_label) or str(std_label).strip() == '':
                    continue  # Skip rows with no valid data
            
            std_label = str(std_label).strip()
            
            for col in dataset_cols:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    # Convert column name back to dataset format
                    dataset = self._unsanitize_dataset_name(col)
                    
                    # Handle semicolon-separated values (legacy format)
                    for neuron_id in str(value).split(';'):
                        neuron_id = neuron_id.strip()
                        if not neuron_id:
                            continue
                        
                        # Try to convert to int if it looks like a bodyId
                        try:
                            if neuron_id.isdigit():
                                neuron_id = int(neuron_id)
                        except:
                            pass
                        
                        if neuron_id not in mapping[std_label][dataset]:
                            mapping[std_label][dataset].append(neuron_id)
    
    def _load_from_json(self, filepath: str, role: str) -> None:
        """
        Load mapping from JSON file.
        
        JSON Format:
            {
                "source_mapping": {  # or "target_mapping"
                    "custom_label": ["label1", "label2"],  # or "std_label" for legacy
                    "dataset1": [["id1", "id2"], ["id3"]],
                    "dataset2": [["id4"], ["id5", "id6"]]
                }
            }
        
        Args:
            filepath: Path to JSON file
            role: 'source' or 'target'
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Mapping file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        key = f'{role}_mapping'
        if key not in data:
            raise ValueError(f"JSON must have '{key}' key. Found: {list(data.keys())}")
        
        mapping_data = data[key]
        
        # Support both 'custom_label' (preferred) and 'std_label' (legacy)
        label_key = None
        if 'custom_label' in mapping_data:
            label_key = 'custom_label'
        elif 'std_label' in mapping_data:
            label_key = 'std_label'
        else:
            raise ValueError(f"JSON {key} must have 'custom_label' or 'std_label' array")
        
        std_labels = mapping_data[label_key]
        mapping = self._source_mapping if role == 'source' else self._target_mapping
        
        # Get dataset keys (all except label key)
        dataset_keys = [k for k in mapping_data.keys() if k != label_key]
        
        for dataset in dataset_keys:
            groups = mapping_data[dataset]
            
            if len(groups) != len(std_labels):
                raise ValueError(f"Dataset '{dataset}' has {len(groups)} groups but {len(std_labels)} std_labels")
            
            for i, (label, group) in enumerate(zip(std_labels, groups)):
                if isinstance(group, list):
                    for neuron_id in group:
                        mapping[label][dataset].append(neuron_id)
                else:
                    mapping[label][dataset].append(group)
    
    def _load_source_from_dict(self, mapping_dict: Dict, labels: List[str]) -> None:
        """
        Load source mapping from dictionary.
        
        Args:
            mapping_dict: {dataset: [[group1_ids], [group2_ids], ...]}
            labels: ['label1', 'label2', ...] matching group count
        """
        self._load_from_dict(mapping_dict, labels, 'source')
    
    def _load_target_from_dict(self, mapping_dict: Dict, labels: List[str]) -> None:
        """
        Load target mapping from dictionary.
        
        Args:
            mapping_dict: {dataset: [[group1_ids], [group2_ids], ...]}
            labels: ['label1', 'label2', ...] matching group count
        """
        self._load_from_dict(mapping_dict, labels, 'target')
    
    def _load_from_dict(self, mapping_dict: Dict, labels: List[str], role: str) -> None:
        """
        Load mapping from dictionary input.
        
        Args:
            mapping_dict: {dataset: [[group1_ids], [group2_ids], ...]} or {dataset: [flat_ids]}
            labels: Standard labels for groups
            role: 'source' or 'target'
        """
        mapping = self._source_mapping if role == 'source' else self._target_mapping
        
        for dataset, neurons in mapping_dict.items():
            # Ensure neurons is a list of groups
            if not neurons:
                continue
            
            # Check if already grouped (list of lists)
            if isinstance(neurons[0], list):
                groups = neurons
            else:
                # Wrap in single group
                groups = [neurons]
            
            # Auto-generate labels if not provided
            if not labels:
                labels = [f'{role}_grp{i+1}' for i in range(len(groups))]
            
            if len(groups) > len(labels):
                # Extend labels if needed
                labels = labels + [f'{role}_grp{i+1}' for i in range(len(labels), len(groups))]
            
            for label, group in zip(labels, groups):
                for neuron_id in group:
                    mapping[label][dataset].append(neuron_id)
    
    def _build_reverse_lookups(self) -> None:
        """Build reverse lookup dictionaries for fast std_label retrieval."""
        # Source reverse lookup
        for std_label, dataset_dict in self._source_mapping.items():
            for dataset, neuron_ids in dataset_dict.items():
                for neuron_id in neuron_ids:
                    self._source_reverse[dataset][str(neuron_id)] = std_label
        
        # Target reverse lookup
        for std_label, dataset_dict in self._target_mapping.items():
            for dataset, neuron_ids in dataset_dict.items():
                for neuron_id in neuron_ids:
                    self._target_reverse[dataset][str(neuron_id)] = std_label
    
    def _sanitize_dataset_name(self, dataset: str) -> str:
        """Convert dataset name to column-safe format."""
        return dataset.replace(':', '_').replace('.', '_').replace('-', '_')
    
    def _unsanitize_dataset_name(self, column: str) -> str:
        """
        Attempt to convert sanitized column name back to dataset format.
        Note: This is a best-effort conversion as some info may be lost.
        """
        # Common patterns
        if column.startswith('hemibrain_v'):
            return column.replace('hemibrain_v', 'hemibrain:v').replace('_', '.', 1)
        elif column.startswith('male_cns_v') or column.startswith('male-cns_v'):
            return column.replace('male_cns_v', 'male-cns:v').replace('_', '.')
        elif 'flywire' in column.lower() or 'fafb' in column.lower():
            return column  # FlyWire names typically don't have colons
        else:
            return column
    
    def get_std_label(self, dataset: str, original_id: Union[str, int], role: str) -> str:
        """
        Get standardized label for a neuron ID.
        
        Args:
            dataset: Dataset identifier
            original_id: Original neuron ID (type name or bodyId)
            role: 'source' or 'target'
            
        Returns:
            Standardized label, or auto-generated if not found
        """
        reverse = self._source_reverse if role == 'source' else self._target_reverse
        str_id = str(original_id)
        
        # Try exact match
        if dataset in reverse and str_id in reverse[dataset]:
            return reverse[dataset][str_id]
        
        # Try sanitized dataset name
        sanitized = self._sanitize_dataset_name(dataset)
        if sanitized in reverse and str_id in reverse[sanitized]:
            return reverse[sanitized][str_id]
        
        # Auto-generate label
        return self.auto_generate_label(original_id)
    
    def get_neurons_for_label(self, std_label: str, dataset: str, role: str) -> List:
        """
        Get all neuron IDs for a standard label in a specific dataset.
        
        Args:
            std_label: Standard label
            dataset: Dataset identifier
            role: 'source' or 'target'
            
        Returns:
            List of neuron IDs (may be empty if dataset doesn't have this label)
        """
        mapping = self._source_mapping if role == 'source' else self._target_mapping
        
        if std_label in mapping:
            if dataset in mapping[std_label]:
                return mapping[std_label][dataset]
            # Try sanitized name
            sanitized = self._sanitize_dataset_name(dataset)
            if sanitized in mapping[std_label]:
                return mapping[std_label][sanitized]
        
        return []
    
    def get_all_std_labels(self, role: str) -> List[str]:
        """
        Get all standard labels for a role.
        
        Args:
            role: 'source' or 'target'
            
        Returns:
            List of all standard labels
        """
        mapping = self._source_mapping if role == 'source' else self._target_mapping
        return list(mapping.keys())
    
    def get_datasets(self, role: str) -> List[str]:
        """
        Get all datasets that have mappings for a role.
        
        Args:
            role: 'source' or 'target'
            
        Returns:
            List of dataset identifiers
        """
        mapping = self._source_mapping if role == 'source' else self._target_mapping
        datasets = set()
        for label_dict in mapping.values():
            datasets.update(label_dict.keys())
        return list(datasets)
    
    @staticmethod
    def auto_generate_label(original_id: Union[str, int]) -> str:
        """
        Auto-generate a standard label for an unmapped ID.
        
        Args:
            original_id: Original neuron ID
            
        Returns:
            Generated label in format {original_id}_etc
        """
        return f"{original_id}_etc"
    
    def get_mapping_summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame of all mappings for user verification.
        
        Returns:
            DataFrame with columns: role, std_label, dataset, neuron_count, neurons
        """
        rows = []
        
        for std_label, dataset_dict in self._source_mapping.items():
            for dataset, neurons in dataset_dict.items():
                rows.append({
                    'role': 'source',
                    'std_label': std_label,
                    'dataset': dataset,
                    'neuron_count': len(neurons),
                    'neurons': str(neurons[:5]) + ('...' if len(neurons) > 5 else '')
                })
        
        for std_label, dataset_dict in self._target_mapping.items():
            for dataset, neurons in dataset_dict.items():
                rows.append({
                    'role': 'target',
                    'std_label': std_label,
                    'dataset': dataset,
                    'neuron_count': len(neurons),
                    'neurons': str(neurons[:5]) + ('...' if len(neurons) > 5 else '')
                })
        
        return pd.DataFrame(rows)
    
    def export_to_parameters(self) -> Dict:
        """
        Export mapping configuration for parameters.json inclusion.
        
        Returns:
            Dictionary with mapping info for reproducibility
        """
        # Convert defaultdicts to regular dicts for JSON serialization
        resolved_source = {}
        for std_label, dataset_dict in self._source_mapping.items():
            resolved_source[std_label] = dict(dataset_dict)
        
        resolved_target = {}
        for std_label, dataset_dict in self._target_mapping.items():
            resolved_target[std_label] = dict(dataset_dict)
        
        return {
            'source_mapping_file': self._source_file,
            'target_mapping_file': self._target_file,
            'resolved_source_mapping': resolved_source,
            'resolved_target_mapping': resolved_target
        }
    
    def has_mapping(self) -> bool:
        """Check if any mappings are defined."""
        return bool(self._source_mapping) or bool(self._target_mapping)
    
    def get_all_neurons_for_dataset(self, dataset: str, role: str) -> List:
        """
        Get all neuron IDs for a dataset in a specific role.
        
        This is used when LabelMapper is passed to ComparisonParameters
        to provide dataset-specific neuron lists.
        
        Args:
            dataset: Dataset identifier
            role: 'source' or 'target'
            
        Returns:
            List of neuron IDs for this dataset
        """
        mapping = self._source_mapping if role == 'source' else self._target_mapping
        
        all_neurons = []
        for std_label, dataset_dict in mapping.items():
            if dataset in dataset_dict:
                all_neurons.extend(dataset_dict[dataset])
            else:
                # Try sanitized name
                sanitized = self._sanitize_dataset_name(dataset)
                if sanitized in dataset_dict:
                    all_neurons.extend(dataset_dict[sanitized])
        
        return all_neurons
    
    def apply_to_dataframe(self, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """
        Apply standard labels to a connection DataFrame.
        
        Adds 'std_label_pre' and 'std_label_post' columns based on type columns.
        
        Args:
            df: DataFrame with type_pre and type_post columns
            dataset: Dataset identifier for this data
            
        Returns:
            DataFrame with added std_label columns
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Apply to pre (source) neurons
        if 'type_pre' in df.columns:
            df['std_label_pre'] = df['type_pre'].apply(
                lambda x: self.get_std_label(dataset, x, 'source') if pd.notna(x) else ''
            )
        elif 'bodyId_pre' in df.columns:
            df['std_label_pre'] = df['bodyId_pre'].apply(
                lambda x: self.get_std_label(dataset, x, 'source')
            )
        
        # Apply to post (target) neurons
        if 'type_post' in df.columns:
            df['std_label_post'] = df['type_post'].apply(
                lambda x: self.get_std_label(dataset, x, 'target') if pd.notna(x) else ''
            )
        elif 'bodyId_post' in df.columns:
            df['std_label_post'] = df['bodyId_post'].apply(
                lambda x: self.get_std_label(dataset, x, 'target')
            )
        
        return df
    
    def __repr__(self) -> str:
        source_labels = len(self._source_mapping)
        target_labels = len(self._target_mapping)
        return f"LabelMapper(source_labels={source_labels}, target_labels={target_labels})"
    
    @staticmethod
    def verify_csv_format(filepath: str) -> Dict[str, Any]:
        """
        Verify CSV mapping file format and return validation results.
        
        Expected CSV Format:
            custom_label,dataset1,dataset2,...
            label1,neuron_id1,neuron_id2,...
            label2,neuron_id3,neuron_id4,...
        
        Args:
            filepath: Path to CSV file to verify
            
        Returns:
            Dict with 'valid', 'errors', 'warnings', and 'summary' keys
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        if not os.path.exists(filepath):
            result['valid'] = False
            result['errors'].append(f"File not found: {filepath}")
            return result
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Failed to read CSV: {str(e)}")
            return result
        
        # Check for required column
        if 'custom_label' not in df.columns and 'std_label' not in df.columns:
            result['valid'] = False
            result['errors'].append(
                "Missing required column: 'custom_label' (or 'std_label' for legacy format). "
                f"Found columns: {df.columns.tolist()}"
            )
            return result
        
        label_col = 'custom_label' if 'custom_label' in df.columns else 'std_label'
        
        # Check for at least one dataset column
        exclude_cols = {'custom_label', 'std_label', 'std_pattern', 'notes', 'description', 'comment'}
        dataset_cols = [c for c in df.columns if c.lower() not in exclude_cols]
        
        if not dataset_cols:
            result['valid'] = False
            result['errors'].append("No dataset columns found. Add columns like 'hemibrain_v1_2_1', 'male_cns_v0_9', etc.")
            return result
        
        # Check for empty labels
        empty_labels = df[label_col].isna().sum()
        if empty_labels > 0:
            result['warnings'].append(f"{empty_labels} rows have empty {label_col} values (will be auto-generated)")
        
        # Check for duplicate labels with different values
        if label_col in df.columns:
            label_counts = df[label_col].value_counts()
            duplicates = label_counts[label_counts > 1]
            if len(duplicates) > 0:
                result['warnings'].append(
                    f"Labels appear multiple times (will be grouped): {duplicates.index.tolist()[:5]}"
                )
        
        # Summary
        result['summary'] = {
            'rows': len(df),
            'label_column': label_col,
            'dataset_columns': dataset_cols,
            'unique_labels': df[label_col].nunique()
        }
        
        return result
    
    @staticmethod
    def verify_json_format(filepath: str) -> Dict[str, Any]:
        """
        Verify JSON mapping file format and return validation results.
        
        Expected JSON Format:
            {
                "source_mapping": {
                    "custom_label": ["label1", "label2"],
                    "dataset1": [["id1", "id2"], ["id3"]],
                    "dataset2": [["id4"], ["id5", "id6"]]
                },
                "target_mapping": {
                    "custom_label": ["label1"],
                    "dataset1": [["id7"]]
                }
            }
        
        Args:
            filepath: Path to JSON file to verify
            
        Returns:
            Dict with 'valid', 'errors', 'warnings', and 'summary' keys
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        if not os.path.exists(filepath):
            result['valid'] = False
            result['errors'].append(f"File not found: {filepath}")
            return result
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            result['valid'] = False
            result['errors'].append(f"Invalid JSON: {str(e)}")
            return result
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Failed to read file: {str(e)}")
            return result
        
        # Check for at least one mapping section
        has_source = 'source_mapping' in data
        has_target = 'target_mapping' in data
        
        if not has_source and not has_target:
            result['valid'] = False
            result['errors'].append(
                "JSON must have 'source_mapping' and/or 'target_mapping' keys. "
                f"Found: {list(data.keys())}"
            )
            return result
        
        summary_sections = {}
        
        # Validate each section
        for section_name in ['source_mapping', 'target_mapping']:
            if section_name not in data:
                continue
            
            section = data[section_name]
            
            # Check for label key
            label_key = None
            if 'custom_label' in section:
                label_key = 'custom_label'
            elif 'std_label' in section:
                label_key = 'std_label'
            
            if not label_key:
                result['valid'] = False
                result['errors'].append(
                    f"'{section_name}' must have 'custom_label' (or 'std_label') array. "
                    f"Found keys: {list(section.keys())}"
                )
                continue
            
            labels = section[label_key]
            if not isinstance(labels, list):
                result['valid'] = False
                result['errors'].append(f"'{section_name}.{label_key}' must be an array")
                continue
            
            # Check dataset arrays
            dataset_keys = [k for k in section.keys() if k not in ['custom_label', 'std_label']]
            
            for dataset_key in dataset_keys:
                dataset_values = section[dataset_key]
                if not isinstance(dataset_values, list):
                    result['valid'] = False
                    result['errors'].append(f"'{section_name}.{dataset_key}' must be an array")
                    continue
                
                if len(dataset_values) != len(labels):
                    result['valid'] = False
                    result['errors'].append(
                        f"'{section_name}.{dataset_key}' has {len(dataset_values)} groups "
                        f"but {len(labels)} labels defined"
                    )
            
            summary_sections[section_name] = {
                'label_key': label_key,
                'label_count': len(labels),
                'datasets': dataset_keys
            }
        
        result['summary'] = summary_sections
        
        return result
    
    @classmethod
    def verify_mapping_file(cls, filepath: str) -> Dict[str, Any]:
        """
        Auto-detect file type and verify format.
        
        Args:
            filepath: Path to CSV or JSON mapping file
            
        Returns:
            Dict with 'valid', 'errors', 'warnings', and 'summary' keys
        """
        if filepath.lower().endswith('.json'):
            return cls.verify_json_format(filepath)
        elif filepath.lower().endswith('.csv'):
            return cls.verify_csv_format(filepath)
        else:
            return {
                'valid': False,
                'errors': [f"Unsupported file type. Use .csv or .json: {filepath}"],
                'warnings': [],
                'summary': {}
            }

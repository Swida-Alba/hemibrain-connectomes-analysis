"""
NeuronFilter - Flexible neuron query parsing utility

This module provides a unified interface for querying neurons using various formats:
- Legacy format: Lists of int (bodyId), string (type), or regex patterns
- Dict-based filter format: Simple filter operators that auto-search columns

The filter format (same as type_filter in NeuronBridge):
- contains: Substring matching
- startswith: Prefix matching  
- endswith: Suffix matching
- regex: Regular expression matching
- exact: Exact value matching (default for lists without operators)

Example usage:
    # Legacy formats (all still supported)
    filter = NeuronFilter(['aMe12', 'Mi1'])  # exact type match
    filter = NeuronFilter(['aMe.*', 'Mi.*'])  # regex patterns
    filter = NeuronFilter([12345, 67890])  # bodyId list
    
    # Dict-based format (same as type_filter)
    filter = NeuronFilter({'contains': 'DN'})  # Auto-searches type, instance, etc.
    filter = NeuronFilter({'startswith': ['aMe', 'Mi']})  # Multiple prefixes (OR)
    filter = NeuronFilter({'endswith': '_R'})  # Suffix match
    filter = NeuronFilter({'contains': 'DN', 'endswith': '_R'})  # AND logic
    filter = NeuronFilter({'regex': r'DN[a-z]\\d+'})  # Regex pattern
    
    # Apply to DataFrame
    matched_df = filter.apply(neuron_df)
    matched_bodyIds = filter.get_bodyIds(neuron_df)
"""

import re
from typing import Union, List, Dict, Any
import pandas as pd


class NeuronFilter:
    """
    Flexible neuron filter supporting legacy list format and dict-based filters.
    
    The dict format uses operators that auto-search across columns (type, instance, bodyId, etc.):
    - {'contains': 'DN'} - Matches neurons where any searchable column contains 'DN'
    - {'startswith': ['aMe', 'Mi']} - Matches prefix in any column (OR for list)
    - {'endswith': '_R'} - Matches suffix
    - {'regex': 'pattern'} - Regex match
    - {'contains': 'DN', 'endswith': '_R'} - AND logic across operators
    
    Search priority: type > instance > bodyId > other string columns
    
    Examples
    --------
    >>> # Legacy formats
    >>> f = NeuronFilter(['aMe12'])  # Exact type match
    >>> f = NeuronFilter(['aMe.*'])  # Regex pattern
    >>> f = NeuronFilter([12345])    # BodyId list
    >>> f = NeuronFilter(None)       # Match all
    
    >>> # Dict-based format (recommended)
    >>> f = NeuronFilter({'contains': 'DN'})
    >>> f = NeuronFilter({'startswith': ['aMe', 'Mi']})
    >>> f = NeuronFilter({'contains': 'DN', 'endswith': '_R'})
    
    >>> # Apply to neuron DataFrame
    >>> matched_df = f.apply(neuron_df)
    >>> bodyIds = f.get_bodyIds(neuron_df)
    """
    
    # Supported filter operators
    OPERATORS = {'contains', 'startswith', 'endswith', 'regex', 'exact', 'not_contains', 'not_regex'}
    
    # Columns to search in order of priority
    SEARCH_COLUMNS = ['type', 'instance', 'bodyId']
    
    def __init__(self, query: Union[None, int, str, List, Dict] = None):
        """
        Initialize NeuronFilter with a query.
        
        Parameters
        ----------
        query : None, int, str, list, or dict
            Neuron query in legacy or dict format.
            
            Legacy formats:
            - None: Match all neurons
            - int: Single bodyId
            - str: Single type/instance (with optional regex pattern)
            - list[int]: List of bodyIds
            - list[str]: List of types/instances (with optional regex patterns)
            
            Dict format (same as type_filter):
            - {'contains': 'DN'}  # Substring match across columns
            - {'startswith': ['aMe', 'Mi']}  # Prefix match (OR for list)
            - {'endswith': '_R'}  # Suffix match
            - {'regex': r'pattern'}  # Regex match
            - {'contains': 'X', 'startswith': 'Y'}  # AND logic
        """
        self.raw_query = query
        self.filter_spec = {}  # {operator: values}
        self.is_legacy = False
        self.match_all = False
        
        self._parse_query(query)
    
    def _parse_query(self, query):
        """Parse query into normalized filter_spec."""
        if query is None:
            self.match_all = True
            return
        
        if isinstance(query, dict):
            self._parse_dict_query(query)
        else:
            self._parse_legacy_query(query)
    
    def _parse_legacy_query(self, query):
        """Parse legacy list/int/string query format."""
        self.is_legacy = True
        
        # Normalize to list
        if not isinstance(query, list):
            query = [query]
        
        if len(query) == 0:
            # Empty list = match all with type
            self.match_all = True
            return
        
        # Determine query type based on first element
        first = query[0]
        
        if isinstance(first, int) or (isinstance(first, str) and first.isdigit()):
            # BodyId list
            bodyIds = [int(x) if isinstance(x, str) else x for x in query]
            self.filter_spec = {'exact': bodyIds}
        else:
            # Type/instance patterns
            patterns = []
            exact_matches = []
            
            for item in query:
                item_str = str(item)
                # Check if it's a regex pattern (contains regex special chars)
                if self._is_regex_pattern(item_str):
                    patterns.append(item_str)
                else:
                    exact_matches.append(item_str)
            
            # Store in filter_spec
            if exact_matches:
                self.filter_spec['exact'] = exact_matches
            if patterns:
                self.filter_spec['regex'] = patterns
    
    def _parse_dict_query(self, query: dict):
        """Parse dict-based query format (flat, like type_filter)."""
        for operator, values in query.items():
            if operator not in self.OPERATORS:
                # Unknown operator - skip with warning
                continue
            
            # Normalize single value to list
            if not isinstance(values, list):
                values = [values]
            
            self.filter_spec[operator] = values
    
    def _is_regex_pattern(self, s: str) -> bool:
        """Check if string contains regex metacharacters."""
        return any(c in s for c in ['.*', '.+', '[', ']', '|', '^', '$', '?', '+', '*', '(', ')'])
    
    def _match_value(self, value: Any, operator: str, patterns: List) -> bool:
        """Check if value matches the filter patterns using the specified operator."""
        if value is None or pd.isna(value):
            return False
        
        value_str = str(value)
        
        if operator == 'exact':
            return value in patterns or value_str in [str(p) for p in patterns]
        elif operator == 'contains':
            return any(str(p) in value_str for p in patterns)
        elif operator == 'not_contains':
            return not any(str(p) in value_str for p in patterns)
        elif operator == 'startswith':
            return any(value_str.startswith(str(p)) for p in patterns)
        elif operator == 'endswith':
            return any(value_str.endswith(str(p)) for p in patterns)
        elif operator == 'regex':
            for pattern in patterns:
                try:
                    if re.search(str(pattern), value_str):
                        return True
                except re.error:
                    # Invalid regex, try exact match
                    if str(pattern) == value_str:
                        return True
            return False
        elif operator == 'not_regex':
            for pattern in patterns:
                try:
                    if re.search(str(pattern), value_str):
                        return False
                except re.error:
                    if str(pattern) == value_str:
                        return False
            return True
        
        return False
    
    def _get_search_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns to search, in priority order."""
        columns = []
        # Priority columns first
        for col in self.SEARCH_COLUMNS:
            if col in df.columns:
                columns.append(col)
        # Add other string columns
        for col in df.columns:
            if col not in columns and df[col].dtype == 'object':
                columns.append(col)
        return columns
    
    def _apply_operator(self, df: pd.DataFrame, operator: str, patterns: List) -> pd.Series:
        """Apply a single operator across all searchable columns (OR across columns)."""
        search_cols = self._get_search_columns(df)
        
        # For exact match with integers (bodyIds), only search bodyId column
        if operator == 'exact' and patterns and isinstance(patterns[0], int):
            if 'bodyId' in df.columns:
                return df['bodyId'].isin(patterns)
            return pd.Series([False] * len(df), index=df.index)
        
        str_patterns = [str(p) for p in patterns]
        # Search across columns with OR logic (per-column matches union).
        # This matches the original per-cell semantics pinned by
        # tests/core/test_audit_fixes.py, including for the negative
        # operators not_contains / not_regex.
        combined_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in search_cols:
            col_mask = self._apply_operator_column(df[col], operator, patterns, str_patterns)
            combined_mask = combined_mask | col_mask
        
        return combined_mask

    def _apply_operator_column(self, series: pd.Series, operator: str,
                               patterns: List, str_patterns: List) -> pd.Series:
        """Vectorized operator application over one column.

        Replicates the per-cell semantics of _match_value exactly:
        - NaN/None values never match
        - values are compared via their str() form (like str(value))
        - regex patterns use re.search semantics; an invalid regex falls
          back to exact equality for that pattern
        """
        notna = series.notna()
        if not patterns:
            return pd.Series([False] * len(series), index=series.index)
        str_series = series.astype(str)

        if operator == 'exact':
            return (series.isin(patterns) | str_series.isin(str_patterns)) & notna
        if operator == 'contains':
            alt = '|'.join(re.escape(p) for p in str_patterns)
            return str_series.str.contains(alt, na=False) & notna
        if operator == 'not_contains':
            alt = '|'.join(re.escape(p) for p in str_patterns)
            return notna & ~str_series.str.contains(alt, na=False)
        if operator == 'startswith':
            return str_series.str.startswith(tuple(str_patterns), na=False) & notna
        if operator == 'endswith':
            return str_series.str.endswith(tuple(str_patterns), na=False) & notna
        if operator == 'regex':
            mask = pd.Series([False] * len(series), index=series.index)
            for p in str_patterns:
                try:
                    mask = mask | str_series.str.contains(p, regex=True, na=False)
                except re.error:
                    # Invalid regex: original semantics try an exact match
                    mask = mask | (str_series == p)
            return mask & notna
        if operator == 'not_regex':
            mask = pd.Series([True] * len(series), index=series.index)
            for p in str_patterns:
                try:
                    mask = mask & ~str_series.str.contains(p, regex=True, na=False)
                except re.error:
                    mask = mask & (str_series != p)
            return mask & notna
        return pd.Series([False] * len(series), index=series.index)
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filter to DataFrame and return matching rows.
        
        Parameters
        ----------
        df : pd.DataFrame
            Neuron DataFrame with columns like bodyId, type, instance
            
        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only matching neurons
        """
        if self.match_all:
            return df.copy()
        
        if len(df) == 0:
            return df.copy()
        
        if not self.filter_spec:
            return df.copy()
        
        # Apply each operator (AND logic across operators; legacy lists with
        # mixed exact+regex groups keep this original semantics too - see
        # tests/core/test_audit_fixes.py::test_legacy_list_semantics_preserved)
        mask = pd.Series([True] * len(df), index=df.index)
        
        for operator, patterns in self.filter_spec.items():
            op_mask = self._apply_operator(df, operator, patterns)
            mask = mask & op_mask
        
        return df[mask].copy()
    
    def get_bodyIds(self, df: pd.DataFrame) -> List:
        """
        Get list of bodyIds matching the filter.
        
        Parameters
        ----------
        df : pd.DataFrame
            Neuron DataFrame with bodyId column
            
        Returns
        -------
        list
            List of matching bodyIds
        """
        matched = self.apply(df)
        if 'bodyId' in matched.columns:
            return matched['bodyId'].tolist()
        return []
    
    def get_types(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of unique types matching the filter.
        
        Parameters
        ----------
        df : pd.DataFrame
            Neuron DataFrame with type column
            
        Returns
        -------
        list
            List of unique type names
        """
        matched = self.apply(df)
        if 'type' in matched.columns:
            return matched['type'].dropna().unique().tolist()
        return []
    
    def get_instances(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of unique instances matching the filter.
        
        Parameters
        ----------
        df : pd.DataFrame
            Neuron DataFrame with instance column
            
        Returns
        -------
        list
            List of unique instance names
        """
        matched = self.apply(df)
        if 'instance' in matched.columns:
            return matched['instance'].dropna().unique().tolist()
        return []
    
    def describe(self) -> str:
        """Return human-readable description of the filter."""
        if self.match_all:
            return "Match all neurons"
        
        if not self.filter_spec:
            return "No filter"
        
        parts = []
        for op, vals in self.filter_spec.items():
            if len(vals) == 1:
                parts.append(f"{op}='{vals[0]}'")
            else:
                parts.append(f"{op}={vals}")
        
        return " AND ".join(parts)
    
    def __repr__(self):
        return f"NeuronFilter({self.describe()})"
    
    def __bool__(self):
        """Return False if match_all (allows `if filter:` checks)."""
        return not self.match_all


def parse_neuron_query(query: Union[None, int, str, List, Dict]) -> NeuronFilter:
    """
    Convenience function to create a NeuronFilter from any query format.
    
    This is the main entry point for parsing neuron queries.
    
    Parameters
    ----------
    query : various
        See NeuronFilter for supported formats
        
    Returns
    -------
    NeuronFilter
        Configured filter object
        
    Examples
    --------
    >>> filter = parse_neuron_query(['aMe.*', 'Mi1'])
    >>> filter = parse_neuron_query({'contains': 'DN'})
    """
    return NeuronFilter(query)


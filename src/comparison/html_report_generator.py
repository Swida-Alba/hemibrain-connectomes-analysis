"""
HTML Report Generator for Cross-Dataset Comparison.

Generates a static HTML report organized by content type (not by threshold).
Structure:
- Overview & Key Findings
- Edge Count Charts (all thresholds)
- Similarity Matrices (all thresholds)  
- Networks (side-by-side per threshold)
- Edge Presence Matrices (per threshold)
- Path Presence Matrices (per threshold)
- Conservation Analysis (per threshold)
- Statistics Tables (per threshold)
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def generate_html_report(
    analyzer,
    dataset_names: List[str],
    thresholds: List[int],
    mode_specific_note: str,
    path_count_data: List[Dict],
    key_findings_per_threshold: Dict[int, Dict]
) -> str:
    """
    Generate a static HTML report organized by content type.
    """
    html_parts = []
    
    # Get nicknames for display (shorter names)
    nicknames = analyzer.parameters.get_dataset_nicknames()
    nickname_map = {d: nicknames[i] for i, d in enumerate(dataset_names)}
    
    # HTML Header with CSS
    html_parts.append(_generate_html_header())
    
    # Report header
    html_parts.append(_generate_report_header(analyzer, dataset_names, thresholds, mode_specific_note, nickname_map))
    
    # Table of Contents
    html_parts.append(_generate_toc(thresholds))
    
    # 1. Summary Section
    html_parts.append(_generate_summary_section(
        analyzer, dataset_names, thresholds, path_count_data, 
        key_findings_per_threshold, nickname_map
    ))
    
    # 1.5. Neuron Counts Section
    html_parts.append(_generate_neuron_counts_section(analyzer, dataset_names, nickname_map))
    
    # 2. Similarity Matrices Section
    html_parts.append(_generate_similarity_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 3. Networks Section
    html_parts.append(_generate_networks_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 4. Edge Presence Matrices Section
    html_parts.append(_generate_edge_matrices_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 5. Path Presence Matrices Section
    html_parts.append(_generate_path_matrices_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 6. Conservation Analysis Section
    html_parts.append(_generate_conservation_section(analyzer, dataset_names, thresholds, 
                                                      key_findings_per_threshold, nickname_map))
    
    # 6.5. Dataset Overlap Matrices Section
    html_parts.append(_generate_overlap_matrices_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 7. Statistics Section
    html_parts.append(_generate_statistics_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # Footer
    html_parts.append(_generate_footer())
    
    return ''.join(html_parts)


def _generate_html_header() -> str:
    """Generate HTML head with CSS styles."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-Dataset Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #22c55e;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --border-color: #e2e8f0;
            --conserved-color: #22c55e;
            --partial-color: #f59e0b;
            --unique-color: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: #1e293b;
            line-height: 1.6;
        }
        .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
        header {
            background: linear-gradient(135deg, var(--primary-color), #1d4ed8);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
        }
        header h1 { font-size: 2rem; margin-bottom: 10px; }
        header p { opacity: 0.9; }
        .section { margin: 40px 0; }
        .section-header {
            background: linear-gradient(90deg, var(--primary-color), #60a5fa);
            color: white;
            padding: 15px 25px;
            border-radius: 12px 12px 0 0;
            font-size: 1.3rem;
            font-weight: 600;
        }
        .section-content {
            background: var(--card-bg);
            border-radius: 0 0 12px 12px;
            padding: 24px;
            border: 1px solid var(--border-color);
            border-top: none;
        }
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }
        .card h3 {
            color: var(--primary-color);
            margin-bottom: 16px;
            font-size: 1.1rem;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }
        .card h4 {
            color: var(--secondary-color);
            margin: 12px 0 8px 0;
            font-size: 1rem;
        }
        .grid { display: grid; gap: 20px; }
        .grid-2 { grid-template-columns: repeat(2, 1fr); }
        .grid-3 { grid-template-columns: repeat(3, 1fr); }
        .stat-box {
            background: var(--bg-color);
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-box .value { font-size: 1.8rem; font-weight: bold; color: var(--primary-color); }
        .stat-box .label { color: var(--secondary-color); font-size: 0.85rem; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.85rem;
        }
        th, td {
            padding: 8px 10px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        th { background: var(--bg-color); font-weight: 600; }
        tr:hover { background: var(--bg-color); }
        /* Sticky header for scrollable tables */
        .sticky-table-container {
            max-height: 500px;
            overflow-y: auto;
            position: relative;
        }
        .sticky-table-container thead th {
            position: sticky;
            top: 0;
            background: #f1f5f9;
            z-index: 10;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .badge {
            padding: 3px 8px;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .badge-success { background: #dcfce7; color: #166534; }
        .badge-warning { background: #fef3c7; color: #92400e; }
        .badge-danger { background: #fee2e2; color: #991b1b; }
        .presence-check { color: var(--success-color); }
        .presence-cross { color: var(--danger-color); }
        .toc {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }
        .toc h2 { margin-bottom: 15px; color: var(--primary-color); }
        .toc ul { list-style: none; display: flex; flex-wrap: wrap; gap: 10px; }
        .toc a {
            display: inline-block;
            padding: 8px 16px;
            background: var(--bg-color);
            color: var(--primary-color);
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .toc a:hover { background: var(--primary-color); color: white; }
        .print-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 24px;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 1000;
        }
        .print-btn:hover { background: #1d4ed8; }
        .tabs { margin-top: 15px; }
        .tab-buttons { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 15px; }
        .tab-btn {
            padding: 8px 16px;
            border: 1px solid var(--border-color);
            background: var(--bg-color);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .tab-btn:hover { background: #e2e8f0; }
        .tab-btn.active { background: var(--primary-color); color: white; border-color: var(--primary-color); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .network-legend {
            display: flex;
            gap: 20px;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-color {
            width: 20px;
            height: 4px;
            border-radius: 2px;
        }
        .chart-container { min-height: 350px; }
        @media print {
            .print-btn { display: none; }
            .card { break-inside: avoid; }
            .section { break-before: page; }
            .tab-content { display: block !important; }
            .tab-buttons { display: none; }
        }
        @media (max-width: 768px) {
            .grid-2, .grid-3 { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
"""


def _generate_report_header(analyzer, dataset_names: List[str], thresholds: List[int], 
                            mode_specific_note: str, nickname_map: Dict[str, str]) -> str:
    """Generate the report header section."""
    from datetime import datetime
    
    params = analyzer.parameters
    thresholds_str = ', '.join(str(t) for t in thresholds)
    datasets_display = ', '.join(nickname_map[d] for d in dataset_names)
    
    return f"""
        <header>
            <h1>📊 Cross-Dataset Comparison Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Datasets: {datasets_display}</p>
            <p>Thresholds: {thresholds_str} | Mode: <strong>{params.comparison_mode}</strong> | Max Interlayer: {params.max_interlayer}</p>
        </header>
        {mode_specific_note}
"""


def _generate_toc(thresholds: List[int]) -> str:
    """Generate table of contents."""
    return """
        <div class="toc">
            <h2>📑 Quick Navigation</h2>
            <ul>
                <li><a href="#summary">📋 Summary & Key Findings</a></li>
                <li><a href="#neuron-counts">🧬 Neuron Counts Comparison</a></li>
                <li><a href="#similarity">🔢 Similarity Matrices</a></li>
                <li><a href="#networks">🕸️ Network Visualizations</a></li>
                <li><a href="#edge-matrices">🔗 Edge Presence Matrices</a></li>
                <li><a href="#path-matrices">🛤️ Path Presence Matrices</a></li>
                <li><a href="#conservation">🏆 Conservation Analysis</a></li>
                <li><a href="#statistics">📉 Statistics</a></li>
                <li><a href="connectivity_profile_comparison.html" target="_blank" style="color: #7c3aed;">🔬 Connectivity Profile Comparison Report ↗</a></li>
            </ul>
        </div>
"""


def _generate_summary_section(analyzer, dataset_names: List[str], thresholds: List[int],
                               path_count_data: List[Dict], key_findings: Dict,
                               nickname_map: Dict[str, str]) -> str:
    """Generate summary section with overview and charts."""
    html_parts = []
    
    html_parts.append("""
        <div id="summary" class="section">
            <div class="section-header">📋 Summary & Key Findings</div>
            <div class="section-content">
""")
    
    # Key findings table
    html_parts.append('<div class="card"><h3>Key Metrics by Threshold</h3>')
    html_parts.append('<table><thead><tr><th>Threshold</th><th>Total Edges</th><th>Conserved Edges</th><th>Edge Rate</th><th>Total Paths</th><th>Conserved Paths</th><th>Path Rate</th></tr></thead><tbody>')
    
    for t in thresholds:
        kf = key_findings.get(t, {})
        total_edges = kf.get('total_edges', 0)
        common_edges = kf.get('common_edges', 0)
        edge_rate = (common_edges / total_edges * 100) if total_edges > 0 else 0
        total_paths = kf.get('total_paths', 0)
        common_paths = kf.get('common_paths', 0)
        path_rate = (common_paths / total_paths * 100) if total_paths > 0 else 0
        
        html_parts.append(f'<tr><td><strong>t={t}</strong></td><td>{total_edges}</td><td>{common_edges}</td><td>{edge_rate:.1f}%</td><td>{total_paths}</td><td>{common_paths}</td><td>{path_rate:.1f}%</td></tr>')
    
    html_parts.append('</tbody></table></div>')
    
    # Edge counts chart
    corrected_data = []
    for d in dataset_names:
        nick = nickname_map[d]
        for t in thresholds:
            aligned = analyzer.get_aligned_data(t)
            count = int((aligned[d] > 0).sum()) if not aligned.empty and d in aligned.columns else 0
            corrected_data.append({'dataset': nick, 'threshold': t, 'count': count})
    
    data_json = json.dumps(corrected_data)
    nicknames_json = json.dumps([nickname_map[d] for d in dataset_names])
    thresholds_json = json.dumps(thresholds)
    
    # Calculate total weights and avg traversal probability
    total_weight_data = []
    avg_prob_data = []
    for d in dataset_names:
        nick = nickname_map[d]
        for t in thresholds:
            aligned = analyzer.get_aligned_data(t)
            total_w = int(aligned[d].sum()) if not aligned.empty and d in aligned.columns else 0
            total_weight_data.append({'dataset': nick, 'threshold': t, 'weight': total_w})
            
            # Get traversal probability data if available
            try:
                prob_data = analyzer._get_prob_data_for_threshold(t)
                if not prob_data.empty and d in prob_data.columns:
                    # Fill NaN with 0 before calculating mean
                    avg_prob = float(prob_data[d].fillna(0).mean())
                    if pd.isna(avg_prob):
                        avg_prob = 0.0
                else:
                    avg_prob = 0.0
            except:
                avg_prob = 0.0
            avg_prob_data.append({'dataset': nick, 'threshold': t, 'prob': avg_prob})
    
    weight_data_json = json.dumps(total_weight_data)
    prob_data_json = json.dumps(avg_prob_data)
    
    html_parts.append(f"""
                <div class="card">
                    <h3>Edge Counts Across All Thresholds</h3>
                    <div id="edgeCountChart" class="chart-container"></div>
                </div>
                <script>
                    (function() {{
                        const data = {data_json};
                        const datasets = {nicknames_json};
                        const thresholds = {thresholds_json};
                        const traces = datasets.map(ds => {{
                            const yVals = thresholds.map(t => {{
                                const item = data.find(d => d.dataset === ds && d.threshold === t);
                                return item ? item.count : 0;
                            }});
                            return {{
                                name: ds,
                                x: thresholds.map(t => 'T=' + t),
                                y: yVals,
                                type: 'bar',
                                text: yVals.map(v => v.toString()),
                                textposition: 'outside'
                            }};
                        }});
                        const allYVals = traces.flatMap(t => t.y);
                        const maxY = Math.max(...allYVals);
                        Plotly.newPlot('edgeCountChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ title: 'Threshold' }},
                            yaxis: {{ title: 'Edge Count', range: [0, maxY * 1.15] }},
                            legend: {{ orientation: 'h', y: -0.15 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
                
                <div class="card">
                    <h3>Total Edge Weight Across All Thresholds</h3>
                    <div id="totalWeightChart" class="chart-container"></div>
                </div>
                <script>
                    (function() {{
                        const data = {weight_data_json};
                        const datasets = {nicknames_json};
                        const thresholds = {thresholds_json};
                        const traces = datasets.map(ds => {{
                            const yVals = thresholds.map(t => {{
                                const item = data.find(d => d.dataset === ds && d.threshold === t);
                                return item ? item.weight : 0;
                            }});
                            return {{
                                name: ds,
                                x: thresholds.map(t => 'T=' + t),
                                y: yVals,
                                type: 'bar',
                                text: yVals.map(v => v.toString()),
                                textposition: 'outside'
                            }};
                        }});
                        const allYVals = traces.flatMap(t => t.y);
                        const maxY = Math.max(...allYVals);
                        Plotly.newPlot('totalWeightChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ title: 'Threshold' }},
                            yaxis: {{ title: 'Total Edge Weight (Synapse Count)', range: [0, maxY * 1.15] }},
                            legend: {{ orientation: 'h', y: -0.15 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
                
                <div class="card">
                    <h3>Average Path Traversal Probability</h3>
                    <div id="avgProbChart" class="chart-container"></div>
                </div>
                <script>
                    (function() {{
                        const data = {prob_data_json};
                        const datasets = {nicknames_json};
                        const thresholds = {thresholds_json};
                        const traces = datasets.map(ds => {{
                            const yVals = thresholds.map(t => {{
                                const item = data.find(d => d.dataset === ds && d.threshold === t);
                                return item ? item.prob : 0;
                            }});
                            return {{
                                name: ds,
                                x: thresholds.map(t => 'T=' + t),
                                y: yVals,
                                type: 'bar',
                                text: yVals.map(v => v.toFixed(4)),
                                textposition: 'outside'
                            }};
                        }});
                        const allYVals = traces.flatMap(t => t.y);
                        const maxY = Math.max(...allYVals);
                        Plotly.newPlot('avgProbChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ title: 'Threshold' }},
                            yaxis: {{ title: 'Avg Traversal Probability', range: [0, maxY * 1.15] }},
                            legend: {{ orientation: 'h', y: -0.15 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
""")
    
    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _generate_neuron_counts_section(analyzer, dataset_names: List[str], 
                                     nickname_map: Dict[str, str]) -> str:
    """Generate neuron counts comparison section."""
    html_parts = []
    
    html_parts.append("""
        <div id="neuron-counts" class="section">
            <div class="section-header">🧬 Neuron Counts Comparison</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Comparison of source/target neuron existence across datasets. Shows which neuron types are present in each dataset.
                </p>
""")
    
    # Get neuron count data from analyzer
    summary_df = getattr(analyzer, '_neuron_counts_summary', None)
    type_df = getattr(analyzer, '_neuron_type_counts', None)
    group_df = getattr(analyzer, '_neuron_group_counts', None)
    
    # Summary table with bar chart
    if summary_df is not None and not summary_df.empty:
        html_parts.append('<div class="card"><h3>Summary: Total Neuron Counts</h3>')
        html_parts.append('<table><thead><tr><th>Dataset</th><th>Source Neurons</th><th>Target Neurons</th><th>Source Types</th><th>Target Types</th></tr></thead><tbody>')
        
        for _, row in summary_df.iterrows():
            html_parts.append(f'''<tr>
                <td><strong>{row.get("dataset", "")}</strong></td>
                <td>{row.get("source_count", 0)}</td>
                <td>{row.get("target_count", 0)}</td>
                <td>{row.get("source_types", 0)}</td>
                <td>{row.get("target_types", 0)}</td>
            </tr>''')
        
        html_parts.append('</tbody></table>')
        
        # Add bar chart for neuron counts
        chart_data = []
        for _, row in summary_df.iterrows():
            chart_data.append({
                'dataset': row.get('dataset', ''),
                'source': row.get('source_count', 0),
                'target': row.get('target_count', 0)
            })
        
        chart_json = json.dumps(chart_data)
        html_parts.append(f'''
            <div id="neuronCountChart" class="chart-container" style="height: 300px;"></div>
            <script>
                (function() {{
                    const data = {chart_json};
                    const datasets = data.map(d => d.dataset);
                    Plotly.newPlot('neuronCountChart', [
                        {{
                            name: 'Source Neurons',
                            x: datasets,
                            y: data.map(d => d.source),
                            type: 'bar',
                            marker: {{ color: '#3b82f6' }}
                        }},
                        {{
                            name: 'Target Neurons', 
                            x: datasets,
                            y: data.map(d => d.target),
                            type: 'bar',
                            marker: {{ color: '#22c55e' }}
                        }}
                    ], {{
                        barmode: 'group',
                        xaxis: {{ title: 'Dataset' }},
                        yaxis: {{ title: 'Neuron Count' }},
                        legend: {{ orientation: 'h', y: -0.15 }}
                    }}, {{responsive: true}});
                }})();
            </script>
        ''')
        html_parts.append('</div>')
    
    # Type counts table - split into source and target tables horizontally
    if type_df is not None and not type_df.empty:
        html_parts.append('<div class="card"><h3>Neuron Counts by Type</h3>')
        html_parts.append('<p style="color: var(--secondary-color); font-size: 0.9em; margin-bottom: 10px;">Shows neuron count per type in each dataset, split by source and target roles.</p>')
        
        # Separate source and target columns
        source_cols = [c for c in type_df.columns if c not in ['type', 'role'] and 'source' in c.lower()]
        target_cols = [c for c in type_df.columns if c not in ['type', 'role'] and 'target' in c.lower()]
        
        # Create two tables side by side
        html_parts.append('<div style="display: flex; gap: 30px; flex-wrap: wrap;">')
        
        # Source table
        if source_cols:
            html_parts.append('<div style="flex: 1; min-width: 300px;"><h4 style="color: #ef4444; margin-bottom: 10px;">🔴 Source Neurons</h4>')
            html_parts.append('<div class="sticky-table-container" style="overflow-x: auto;"><table class="presence-table"><thead><tr><th>Type</th>')
            for col in source_cols:
                # Clean column name - remove "_source" suffix
                display_col = col.replace('_source', '').replace('_v', ':v').replace('_', ' ')
                html_parts.append(f'<th>{display_col}</th>')
            html_parts.append('</tr></thead><tbody>')
            
            for i, (_, row) in enumerate(type_df.iterrows()):
                if i >= 50:
                    html_parts.append(f'<tr><td colspan="{len(source_cols)+1}"><em>... and {len(type_df)-50} more types</em></td></tr>')
                    break
                # Check if this row has any source values
                has_values = any(row.get(col, 0) > 0 for col in source_cols if not pd.isna(row.get(col, 0)))
                if not has_values:
                    continue
                html_parts.append(f'<tr><td><strong>{row.get("type", "")}</strong></td>')
                for col in source_cols:
                    val = row.get(col, 0)
                    if pd.isna(val) or val == 0:
                        html_parts.append('<td class="absent">-</td>')
                    else:
                        html_parts.append(f'<td class="present">{int(val)}</td>')
                html_parts.append('</tr>')
            
            html_parts.append('</tbody></table></div></div>')
        
        # Target table
        if target_cols:
            html_parts.append('<div style="flex: 1; min-width: 300px;"><h4 style="color: #8b5cf6; margin-bottom: 10px;">🟣 Target Neurons</h4>')
            html_parts.append('<div class="sticky-table-container" style="overflow-x: auto;"><table class="presence-table"><thead><tr><th>Type</th>')
            for col in target_cols:
                # Clean column name - remove "_target" suffix
                display_col = col.replace('_target', '').replace('_v', ':v').replace('_', ' ')
                html_parts.append(f'<th>{display_col}</th>')
            html_parts.append('</tr></thead><tbody>')
            
            for i, (_, row) in enumerate(type_df.iterrows()):
                if i >= 50:
                    html_parts.append(f'<tr><td colspan="{len(target_cols)+1}"><em>... and {len(type_df)-50} more types</em></td></tr>')
                    break
                # Check if this row has any target values
                has_values = any(row.get(col, 0) > 0 for col in target_cols if not pd.isna(row.get(col, 0)))
                if not has_values:
                    continue
                html_parts.append(f'<tr><td><strong>{row.get("type", "")}</strong></td>')
                for col in target_cols:
                    val = row.get(col, 0)
                    if pd.isna(val) or val == 0:
                        html_parts.append('<td class="absent">-</td>')
                    else:
                        html_parts.append(f'<td class="present">{int(val)}</td>')
                html_parts.append('</tr>')
            
            html_parts.append('</tbody></table></div></div>')
        
        html_parts.append('</div></div>')  # Close flex container and card
    
    # Group counts table (if custom groups exist)
    if group_df is not None and not group_df.empty:
        html_parts.append('<div class="card"><h3>Neuron Counts by Custom Group</h3>')
        html_parts.append('<div style="overflow-x: auto;"><table class="presence-table"><thead><tr><th>Custom Group</th>')
        
        data_cols = [c for c in group_df.columns if c not in ['custom_group', 'role']]
        for col in data_cols:
            display_col = col.replace('_v', ':v').replace('_', ' ')
            html_parts.append(f'<th>{display_col}</th>')
        html_parts.append('</tr></thead><tbody>')
        
        for _, row in group_df.iterrows():
            html_parts.append(f'<tr><td><strong>{row.get("custom_group", "")}</strong></td>')
            for col in data_cols:
                val = row.get(col, 0)
                if pd.isna(val) or val == 0:
                    html_parts.append('<td class="absent">-</td>')
                else:
                    html_parts.append(f'<td class="present">{int(val)}</td>')
            html_parts.append('</tr>')
        
        html_parts.append('</tbody></table></div></div>')
    
    # If no data available, show message
    if (summary_df is None or summary_df.empty) and (type_df is None or type_df.empty):
        html_parts.append('''
            <div class="card">
                <p style="color: var(--secondary-color);">
                    ℹ️ Neuron count data not available. Run export_results() to generate this comparison.
                </p>
            </div>
        ''')
    
    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _generate_similarity_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                  nickname_map: Dict[str, str]) -> str:
    """Generate similarity matrices section with square cell heatmaps for all metrics."""
    from .metrics import ComparisonMetrics
    metrics = ComparisonMetrics()
    
    html_parts = []
    html_parts.append("""
        <div id="similarity" class="section">
            <div class="section-header">🔢 Similarity Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Pairwise similarity between datasets at each threshold.<br>
                    <strong>TOPOLOGY METRICS</strong> (binary edge presence):<br>
                    &nbsp;&nbsp;• <strong>Jaccard</strong> = edge set overlap |A∩B|/|A∪B|<br>
                    &nbsp;&nbsp;• <strong>GED</strong> = Graph Edit Distance similarity<br>
                    <strong>MATRIX-BASED METRICS</strong> (uses normalized edge weights):<br>
                    &nbsp;&nbsp;• <strong>Spearman</strong> = rank correlation of shared edge weights<br>
                    &nbsp;&nbsp;• <strong>RV Coefficient</strong> = multivariate matrix similarity
                </p>
""")
    
    # Note: Connectivity Profile Similarity is shown in the separate connectivity_profile_comparison.html
    # report since it uses different methodology (graph-based rank correlation vs threshold-sensitive edge comparison)
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        if aligned.empty:
            continue
        
        available = [d for d in dataset_names if d in aligned.columns]
        if len(available) < 2:
            continue
        
        labels = [nickname_map[d] for d in available]
        similarities = metrics.calculate_all_pairwise_similarities(aligned, dataset_names, threshold=1, include_advanced_metrics=True)
        
        # Save similarity data to CSV (inside comparison_results folder)
        try:
            full_output_path = getattr(analyzer.parameters, 'full_output_path', None)
            if full_output_path:
                csv_dir = os.path.join(full_output_path, 'similarity_matrices')
                os.makedirs(csv_dir, exist_ok=True)
                similarities.to_csv(os.path.join(csv_dir, f'similarity_threshold_{threshold}.csv'), index=False)
        except Exception:
            pass
        
        n = len(available)
        # Initialize similarity matrices for 6 metrics (removed frobenius and pearson)
        jaccard = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        ruzicka = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        spearman_sim = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        rv_sim = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        ged_sim = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        kernel_sim = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        
        for _, row in similarities.iterrows():
            d1, d2 = row['dataset_1'], row['dataset_2']
            if d1 in available and d2 in available:
                i1, i2 = available.index(d1), available.index(d2)
                jac = row.get('jaccard_similarity', 0)
                ruz = row.get('ruzicka_similarity', 0)
                # Use new column name: spearman_rank_correlation (was spearman_correlation)
                spearman_val = row.get('spearman_rank_correlation', 0)
                rv_val = row.get('rv_coefficient', 0)
                ged_val = row.get('ged_similarity', 0)
                kernel_val = row.get('kernel_similarity', 0)
                # Handle NaN values
                if pd.isna(jac): jac = 0
                if pd.isna(ruz): ruz = 0
                if pd.isna(spearman_val): spearman_val = 0
                if pd.isna(rv_val): rv_val = 0
                if pd.isna(ged_val): ged_val = 0
                if pd.isna(kernel_val): kernel_val = 0
                # Fill symmetric matrices
                jaccard[i1][i2] = jaccard[i2][i1] = jac
                ruzicka[i1][i2] = ruzicka[i2][i1] = ruz
                spearman_sim[i1][i2] = spearman_sim[i2][i1] = spearman_val
                rv_sim[i1][i2] = rv_sim[i2][i1] = rv_val
                ged_sim[i1][i2] = ged_sim[i2][i1] = ged_val
                kernel_sim[i1][i2] = kernel_sim[i2][i1] = kernel_val
        
        # Calculate cell size for square cells - smaller to fit 4 in a row
        cell_size = 50
        # Chart size scales with number of datasets but caps for 4-in-row layout
        chart_size = min(n * cell_size + 80, 200)
        
        # Display all 4 metrics in a single row with responsive layout
        html_parts.append(f"""
                <div class="card">
                    <h3>Threshold = {threshold}</h3>
                    
                    <!-- All 4 metrics in one row -->
                    <div style="display: flex; flex-wrap: nowrap; gap: 10px; overflow-x: auto; padding: 8px 0;">
                        <!-- Jaccard -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: 25%; background: #eff6ff; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #1e40af; text-align: center;">🔷 Jaccard</h5>
                            <div id="jaccard_{threshold}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                        <!-- GED -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: 25%; background: #eff6ff; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #1e40af; text-align: center;">🔷 GED</h5>
                            <div id="ged_{threshold}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                        <!-- Spearman -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: 25%; background: #fef3c7; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #92400e; text-align: center;">🔶 Spearman</h5>
                            <div id="spearman_{threshold}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                        <!-- RV -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: 25%; background: #fef3c7; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #92400e; text-align: center;">🔶 RV Coef</h5>
                            <div id="rv_{threshold}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                    </div>
                    <p style="font-size: 0.75em; color: #64748b; margin-top: 8px; text-align: center;">
                        🔷 Topology-based (binary edge presence) | 🔶 Matrix-based (weight-sensitive)
                    </p>
                </div>
                <script>
                    (function() {{
                        const labels = {json.dumps(labels)};
                        const jaccard = {json.dumps(jaccard)};
                        const spearmanSim = {json.dumps(spearman_sim)};
                        const rvSim = {json.dumps(rv_sim)};
                        const gedSim = {json.dumps(ged_sim)};
                        const layout = {{
                            margin: {{ l: 45, r: 10, t: 10, b: 45 }},
                            xaxis: {{ tickangle: -45, scaleanchor: 'y', constrain: 'domain', tickfont: {{size: 8}} }},
                            yaxis: {{ autorange: 'reversed', constrain: 'domain', tickfont: {{size: 8}} }}
                        }};
                        const makeAnnotations = (data, labels) => data.flatMap((row, i) => 
                            row.map((val, j) => ({{
                                x: labels[j], y: labels[i], text: val.toFixed(2), showarrow: false,
                                font: {{ color: val > 0.5 ? 'white' : 'black', size: 10 }}
                            }})));
                        // Use consistent green colorscale: higher value = darker green
                        const greenScale = [[0, '#ffffff'], [0.3, '#c6efce'], [0.6, '#22c55e'], [1, '#166534']];
                        // Topology metrics
                        Plotly.newPlot('jaccard_{threshold}', [{{
                            z: jaccard, x: labels, y: labels, type: 'heatmap',
                            colorscale: greenScale, zmin: 0, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeAnnotations(jaccard, labels)}}, {{responsive: true}});
                        Plotly.newPlot('ged_{threshold}', [{{
                            z: gedSim, x: labels, y: labels, type: 'heatmap',
                            colorscale: greenScale, zmin: 0, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeAnnotations(gedSim, labels)}}, {{responsive: true}});
                        // Matrix-based metrics
                        Plotly.newPlot('spearman_{threshold}', [{{
                            z: spearmanSim, x: labels, y: labels, type: 'heatmap',
                            colorscale: greenScale, zmin: 0, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeAnnotations(spearmanSim, labels)}}, {{responsive: true}});
                        Plotly.newPlot('rv_{threshold}', [{{
                            z: rvSim, x: labels, y: labels, type: 'heatmap',
                            colorscale: greenScale, zmin: 0, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeAnnotations(rvSim, labels)}}, {{responsive: true}});
                    }})();
                </script>
""")
    
    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _generate_networks_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                nickname_map: Dict[str, str]) -> str:
    """Generate networks section with conservation-colored edges and role-colored nodes."""
    thresholds_json = json.dumps(thresholds)
    num_networks = len(thresholds)
    # Responsive grid: 1 col on small, 2 cols if 2+ networks
    grid_cols = min(num_networks, 2)
    
    # Check if source == target (self-edge scenario)
    is_source_equals_target = False
    self_edge_warning = ""
    try:
        src_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.source_neurons)
        tgt_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.target_neurons)
        if set(src_list) == set(tgt_list):
            is_source_equals_target = True
            # Count self-edges across all thresholds
            self_edge_count = 0
            for threshold in thresholds:
                aligned = analyzer.get_aligned_data(threshold)
                if not aligned.empty:
                    for edge_key in aligned.index:
                        if ' -> ' in str(edge_key):
                            parts = str(edge_key).split(' -> ')
                            if len(parts) == 2 and parts[0] == parts[1]:
                                self_edge_count += 1
                    break  # Only count once (same edges at all thresholds)
            if self_edge_count > 0:
                self_edge_warning = f'''
                <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                    <span style="color: #92400e;">⚠️ <strong>Self-edges detected:</strong> Source and target neuron lists are identical. 
                    <strong>{self_edge_count}</strong> self-edges (type→same type, e.g., aMe1→aMe1) exist in the edge matrix 
                    but are excluded from path analysis (FindAllPath requires source≠target for direct connections).</span>
                </div>'''
    except:
        pass
    
    html_parts = []
    html_parts.append(f"""
        <div id="networks" class="section">
            <div class="section-header">🕸️ Network Visualizations</div>
            <div class="section-content">
                {self_edge_warning}
                <div style="margin-bottom: 15px;">
                    <p style="color: var(--secondary-color); margin: 0 0 8px 0;">
                        <strong>Edge conservation:</strong> 
                        <span style="color: #22c55e;">● Conserved (all)</span> 
                        <span style="color: #f59e0b; margin-left: 12px;">● Partial (some)</span>
                        <span style="color: #94a3b8; margin-left: 12px;">● Unique (one)</span>
                    </p>
                    <p style="color: var(--secondary-color); margin: 0;">
                        <strong>Node role:</strong> 
                        <span style="color: #ef4444;">● Source</span> 
                        <span style="color: #3b82f6; margin-left: 12px;">● Intermediate</span>
                        <span style="color: #8b5cf6; margin-left: 12px;">● Target</span>
                        <span style="color: #9ca3af; margin-left: 12px;">● Dead-end (no path through)</span>
                    </p>
                </div>
                <div style="display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 15px;">
                    <button id="global_conserved_btn" onclick="toggleConservedOnly()" 
                        style="padding: 8px 16px; border-radius: 6px; border: 1px solid var(--border-color); 
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 13px; white-space: nowrap;">
                        🌐 Show All
                    </button>
                    <button id="global_deadend_btn" onclick="toggleDeadEndNodes()" 
                        style="padding: 8px 16px; border-radius: 6px; border: 1px solid var(--border-color); 
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 13px; white-space: nowrap;">
                        👁️ Show Dead-ends
                    </button>
                    <button id="global_physics_btn" onclick="toggleAllNetworks()" 
                        style="padding: 8px 16px; border-radius: 6px; border: 1px solid var(--border-color); 
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 13px; white-space: nowrap;">
                        📌 Static Mode
                    </button>
                </div>
                <script>
                    // Global network mode state (default: static)
                    window.networkPhysicsEnabled = false;
                    window.conservedOnlyMode = false;
                    window.hideDeadEndNodes = false;
                    window.allNetworks = {{}};
                    window.allThresholds = {thresholds_json};
                    
                    function toggleDeadEndNodes() {{
                        window.hideDeadEndNodes = !window.hideDeadEndNodes;
                        const btn = document.getElementById('global_deadend_btn');
                        
                        if (window.hideDeadEndNodes) {{
                            btn.innerHTML = '🚫 Hide Dead-ends';
                            btn.style.background = '#f59e0b';
                        }} else {{
                            btn.innerHTML = '👁️ Show Dead-ends';
                            btn.style.background = 'var(--secondary-color)';
                        }}
                        
                        // Update all networks
                        window.allThresholds.forEach(t => {{
                            if (window.allNetworks[t]) {{
                                const netData = window.allNetworks[t];
                                const net = netData.network;
                                const edges = netData.edges;
                                const nodes = netData.nodes;
                                const allEdges = netData.allEdges;
                                const allNodes = netData.allNodes;
                                const deadEndNodeIds = netData.deadEndNodeIds || new Set();
                                const conservedEdgeIds = netData.conservedEdgeIds;
                                
                                // Determine which edges to show based on both modes
                                let filteredEdges = allEdges;
                                if (window.conservedOnlyMode) {{
                                    filteredEdges = allEdges.filter(e => conservedEdgeIds.has(e.id));
                                }}
                                
                                // Get connected nodes from filtered edges
                                let connectedNodeIds = new Set();
                                filteredEdges.forEach(e => {{
                                    connectedNodeIds.add(e.from);
                                    connectedNodeIds.add(e.to);
                                }});
                                
                                // Filter nodes
                                let filteredNodes;
                                if (window.conservedOnlyMode) {{
                                    filteredNodes = allNodes.filter(n => connectedNodeIds.has(n.id));
                                }} else {{
                                    filteredNodes = allNodes;
                                }}
                                
                                // Apply dead-end filter if enabled
                                if (window.hideDeadEndNodes) {{
                                    filteredNodes = filteredNodes.filter(n => !deadEndNodeIds.has(n.id));
                                    // Also filter edges to/from dead-end nodes
                                    filteredEdges = filteredEdges.filter(e => 
                                        !deadEndNodeIds.has(e.from) && !deadEndNodeIds.has(e.to)
                                    );
                                }}
                                
                                // Apply to network
                                edges.clear();
                                edges.add(filteredEdges);
                                nodes.clear();
                                nodes.add(filteredNodes);
                                
                                // Reinitialize layout
                                net.setOptions({{
                                    nodes: {{ size: 18, font: {{ size: 11 }} }},
                                    edges: {{ smooth: {{ type: 'curvedCW', roundness: 0.1 }} }},
                                    layout: {{
                                        hierarchical: {{
                                            enabled: true,
                                            direction: 'UD',
                                            sortMethod: 'directed',
                                            levelSeparation: 120,
                                            nodeSpacing: 80,
                                            treeSpacing: 100
                                        }}
                                    }},
                                    physics: {{ enabled: false }}
                                }});
                                setTimeout(() => {{
                                    net.setOptions({{ layout: {{ hierarchical: false }} }});
                                    net.fit({{ animation: true }});
                                }}, 200);
                            }}
                        }});
                    }}
                    
                    function toggleConservedOnly() {{
                        window.conservedOnlyMode = !window.conservedOnlyMode;
                        const btn = document.getElementById('global_conserved_btn');
                        
                        if (window.conservedOnlyMode) {{
                            btn.innerHTML = '✅ Conserved Only';
                            btn.style.background = '#22c55e';
                        }} else {{
                            btn.innerHTML = '🌐 Show All';
                            btn.style.background = 'var(--secondary-color)';
                        }}
                        
                        // Update all networks
                        window.allThresholds.forEach(t => {{
                            if (window.allNetworks[t]) {{
                                const netData = window.allNetworks[t];
                                const net = netData.network;
                                const edges = netData.edges;
                                const nodes = netData.nodes;
                                const allEdges = netData.allEdges;
                                const allNodes = netData.allNodes;
                                const conservedEdgeIds = netData.conservedEdgeIds;
                                
                                if (window.conservedOnlyMode) {{
                                    // Filter to conserved edges only
                                    const conservedEdges = allEdges.filter(e => conservedEdgeIds.has(e.id));
                                    edges.clear();
                                    edges.add(conservedEdges);
                                    
                                    // Filter nodes to only those connected by conserved edges
                                    const connectedNodeIds = new Set();
                                    conservedEdges.forEach(e => {{
                                        connectedNodeIds.add(e.from);
                                        connectedNodeIds.add(e.to);
                                    }});
                                    const connectedNodes = allNodes.filter(n => connectedNodeIds.has(n.id));
                                    nodes.clear();
                                    nodes.add(connectedNodes);
                                }} else {{
                                    // Restore all edges and nodes
                                    edges.clear();
                                    edges.add(allEdges);
                                    nodes.clear();
                                    nodes.add(allNodes);
                                }}
                                
                                // Reinitialize layout with hierarchical then disable for free movement
                                // Use 'UD' direction for top-to-bottom flow (consistent with initial layout)
                                net.setOptions({{
                                    nodes: {{ size: 18, font: {{ size: 11 }} }},
                                    edges: {{ smooth: {{ type: 'curvedCW', roundness: 0.1 }} }},
                                    layout: {{
                                        hierarchical: {{
                                            enabled: true,
                                            direction: 'UD',
                                            sortMethod: 'directed',
                                            levelSeparation: 120,
                                            nodeSpacing: 80,
                                            treeSpacing: 100
                                        }}
                                    }},
                                    physics: {{ enabled: false }}
                                }});
                                setTimeout(() => {{
                                    net.setOptions({{ layout: {{ hierarchical: false }} }});
                                    net.fit({{ animation: true }});
                                }}, 200);
                            }}
                        }});
                    }}
                    
                    function toggleAllNetworks() {{
                        window.networkPhysicsEnabled = !window.networkPhysicsEnabled;
                        const btn = document.getElementById('global_physics_btn');
                        
                        if (window.networkPhysicsEnabled) {{
                            btn.innerHTML = '💥 Duang Mode';
                            btn.style.background = 'var(--primary-color)';
                        }} else {{
                            btn.innerHTML = '📌 Static Mode';
                            btn.style.background = 'var(--secondary-color)';
                        }}
                        
                        // Update all networks
                        window.allThresholds.forEach(t => {{
                            if (window.allNetworks[t]) {{
                                const net = window.allNetworks[t].network;
                                
                                if (window.networkPhysicsEnabled) {{
                                    // Duang mode - forceAtlas2Based physics with proper edge length and node sizing
                                    net.setOptions({{
                                        nodes: {{
                                            size: 20,
                                            font: {{ size: 12 }}
                                        }},
                                        edges: {{
                                            smooth: {{ type: 'continuous', roundness: 0.2 }}
                                        }},
                                        layout: {{ hierarchical: false }},
                                        physics: {{
                                            enabled: true,
                                            solver: 'forceAtlas2Based',
                                            forceAtlas2Based: {{
                                                gravitationalConstant: -80,
                                                centralGravity: 0.005,
                                                springLength: 120,
                                                springConstant: 0.06,
                                                damping: 0.5,
                                                avoidOverlap: 0.8
                                            }},
                                            stabilization: {{
                                                enabled: true,
                                                iterations: 100,
                                                updateInterval: 25
                                            }},
                                            minVelocity: 0.5
                                        }}
                                    }});
                                    // Fit after stabilization
                                    net.once('stabilized', () => {{
                                        net.fit({{ animation: true }});
                                    }});
                                }} else {{
                                    // Static mode - re-layout with hierarchical then disable physics for free drag
                                    // Use 'UD' direction for top-to-bottom flow (consistent with initial layout)
                                    net.setOptions({{
                                        nodes: {{
                                            size: 18,
                                            font: {{ size: 11 }}
                                        }},
                                        edges: {{
                                            smooth: {{ type: 'curvedCW', roundness: 0.1 }}
                                        }},
                                        layout: {{
                                            hierarchical: {{
                                                enabled: true,
                                                direction: 'UD',
                                                sortMethod: 'directed',
                                                levelSeparation: 120,
                                                nodeSpacing: 80,
                                                treeSpacing: 100
                                            }}
                                        }},
                                        physics: {{ enabled: false }}
                                    }});
                                    // After applying hierarchical, disable it for free movement
                                    setTimeout(() => {{
                                        net.setOptions({{ layout: {{ hierarchical: false }} }});
                                        net.fit({{ animation: true }});
                                    }}, 200);
                                }}
                            }}
                        }});
                    }}
                </script>
""")
    
    # Networks in single column for proper display
    html_parts.append('<div style="display: flex; flex-direction: column; gap: 20px;">')
    for threshold in thresholds:
        html_parts.append(_generate_conservation_network(analyzer, dataset_names, threshold, nickname_map))
    html_parts.append('</div>')
    
    html_parts.append("""
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_conservation_network(analyzer, dataset_names: List[str], threshold: int,
                                    nickname_map: Dict[str, str]) -> str:
    """Generate network with conservation-based edge coloring and role-based node coloring."""
    aligned = analyzer.get_aligned_data(threshold)
    
    if aligned.empty:
        return '<div class="card"><p>No network data at this threshold.</p></div>'
    
    available = [d for d in dataset_names if d in aligned.columns]
    if not available:
        return '<div class="card"><p>No datasets available.</p></div>'
    
    num_datasets = len(available)
    nicknames = [nickname_map[d] for d in available]
    
    # Get source and target neurons from parameters
    source_neurons = set()
    target_neurons = set()
    try:
        # Get flat list of source/target neuron patterns
        src_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.source_neurons)
        tgt_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.target_neurons)
        source_neurons = set(src_list)
        target_neurons = set(tgt_list)
    except:
        pass
    
    # Build nodes and edges with conservation coloring
    nodes = []
    edges = []
    node_ids = {}
    node_roles = {}  # Track role for each node label
    node_counter = 0
    edge_data = {}  # {(source, target): {dataset: weight}}
    
    # Track incoming and outgoing edges for dead-end detection
    has_outgoing = set()  # nodes that have at least one outgoing edge
    has_incoming = set()  # nodes that have at least one incoming edge
    
    # Collect edge data per dataset
    for edge_key, row in aligned.iterrows():
        if ' -> ' not in str(edge_key):
            continue
        parts = str(edge_key).split(' -> ')
        source, target = parts[0], parts[1] if len(parts) > 1 else ''
        
        edge_tuple = (source, target)
        if edge_tuple not in edge_data:
            edge_data[edge_tuple] = {}
        
        for dataset in available:
            weight = row.get(dataset, 0)
            if weight > 0:
                edge_data[edge_tuple][dataset] = int(weight)
                has_outgoing.add(source)
                has_incoming.add(target)
    
    if not edge_data:
        return '<div class="card"><p>No connections at this threshold.</p></div>'
    
    # Helper function to check if a node label matches any pattern
    def matches_patterns(label: str, patterns: set) -> bool:
        import re
        for pattern in patterns:
            # Convert glob-style pattern to regex
            regex_pattern = pattern.replace('.', r'\.').replace('*', '.*')
            if re.match(f'^{regex_pattern}$', label, re.IGNORECASE):
                return True
            # Also check exact match
            if label == pattern:
                return True
        return False
    
    # First pass: collect all nodes and their initial roles (intermediate)
    all_node_labels = set()
    for (source, target) in edge_data.keys():
        all_node_labels.add(source)
        all_node_labels.add(target)
    
    # Initialize all nodes as intermediate
    for label in all_node_labels:
        node_roles[label] = 'intermediate'
    
    # Override with source/target roles (priority: source > target > intermediate)
    for label in all_node_labels:
        if matches_patterns(label, target_neurons):
            node_roles[label] = 'target'
    for label in all_node_labels:
        if matches_patterns(label, source_neurons):
            node_roles[label] = 'source'
    
    # Detect dead-end nodes: intermediate nodes that only have incoming OR only outgoing edges
    # (i.e., they're not on a complete path from source to target)
    dead_end_nodes = set()
    for label in all_node_labels:
        role = node_roles.get(label, 'intermediate')
        if role == 'intermediate':
            # Dead-end if only incoming (sink) or only outgoing (orphan source in middle)
            only_incoming = label in has_incoming and label not in has_outgoing
            only_outgoing = label in has_outgoing and label not in has_incoming
            if only_incoming or only_outgoing:
                dead_end_nodes.add(label)
    
    # Node colors by role
    role_colors = {
        'source': {'background': '#ef4444', 'border': '#b91c1c'},      # Red
        'intermediate': {'background': '#3b82f6', 'border': '#1d4ed8'}, # Blue
        'target': {'background': '#8b5cf6', 'border': '#6d28d9'},      # Purple
        'dead-end': {'background': '#9ca3af', 'border': '#6b7280'}     # Gray for dead-ends
    }
    
    # Create nodes and edges
    edge_id = 0
    conserved_edge_ids = []  # Track IDs of conserved edges
    conserved_node_ids = set()  # Track node IDs that are part of conserved edges
    
    for (source, target), weights in edge_data.items():
        # Add nodes with role-based coloring
        if source not in node_ids:
            node_ids[source] = node_counter
            role = node_roles.get(source, 'intermediate')
            is_dead_end = source in dead_end_nodes
            if is_dead_end:
                colors = role_colors['dead-end']
                display_role = 'dead-end'
            else:
                colors = role_colors[role]
                display_role = role
            nodes.append({
                'id': node_counter, 
                'label': source, 
                'title': f"{source} ({display_role})",
                'color': colors
            })
            node_counter += 1
        if target not in node_ids:
            node_ids[target] = node_counter
            role = node_roles.get(target, 'intermediate')
            is_dead_end = target in dead_end_nodes
            if is_dead_end:
                colors = role_colors['dead-end']
                display_role = 'dead-end'
            else:
                colors = role_colors[role]
                display_role = role
            nodes.append({
                'id': node_counter, 
                'label': target, 
                'title': f"{target} ({display_role})",
                'color': colors
            })
            node_counter += 1
        
        # Determine conservation level
        present_count = len(weights)
        is_conserved = (present_count == num_datasets)
        if is_conserved:
            color = '#22c55e'  # Conserved - green
            conservation = 'Conserved'
            conserved_edge_ids.append(edge_id)
            # Track nodes that are part of conserved edges
            conserved_node_ids.add(node_ids[source])
            conserved_node_ids.add(node_ids[target])
        elif present_count > 1:
            color = '#f59e0b'  # Partial - orange
            conservation = 'Partial'
        else:
            color = '#94a3b8'  # Unique - gray
            conservation = 'Unique'
        
        # Build hover label with all weights
        hover_lines = [f"{source} → {target}", f"Conservation: {conservation} ({present_count}/{num_datasets})"]
        for i, d in enumerate(available):
            w = weights.get(d, 0)
            status = f"{int(w)}" if w > 0 else "—"
            hover_lines.append(f"{nicknames[i]}: {status}")
        
        edges.append({
            'id': edge_id,
            'from': node_ids[source],
            'to': node_ids[target],
            'color': {'color': color, 'highlight': color},
            'width': 2 + min(present_count, 3),
            'title': '\n'.join(hover_lines)  # Real newline for vis.js tooltip
        })
        edge_id += 1
    
    div_id = f"network_{threshold}"
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    conserved_ids_json = json.dumps(conserved_edge_ids)
    conserved_node_ids_json = json.dumps(list(conserved_node_ids))
    dead_end_node_ids = [node_ids[label] for label in dead_end_nodes if label in node_ids]
    dead_end_node_ids_json = json.dumps(dead_end_node_ids)
    
    # Count dead-end nodes and conserved edges for display
    dead_end_count = len(dead_end_nodes)
    dead_end_info = f' | <span style="color: #9ca3af;"><strong>{dead_end_count}</strong> dead-end</span>' if dead_end_count > 0 else ''
    conserved_count = len(conserved_edge_ids)
    conserved_info = f' | <span style="color: #22c55e;"><strong>{conserved_count}</strong> conserved</span>'
    
    return f'''
        <div class="card">
            <h3>Network at Threshold = {threshold}</h3>
            <div style="color: var(--secondary-color); margin-bottom: 10px;">
                <strong>{len(nodes)}</strong> neurons | <strong>{len(edges)}</strong> unique edges{conserved_info}{dead_end_info}
            </div>
            <div id="{div_id}" style="height: 450px; border: 1px solid var(--border-color); border-radius: 8px;"></div>
        </div>
        <script>
            (function() {{
                const allNodes = {nodes_json};
                const allEdges = {edges_json};
                const conservedEdgeIds = new Set({conserved_ids_json});
                const conservedNodeIds = new Set({conserved_node_ids_json});
                const deadEndNodeIds = new Set({dead_end_node_ids_json});
                
                const nodes = new vis.DataSet(allNodes);
                const edges = new vis.DataSet(allEdges);
                const container = document.getElementById('{div_id}');
                const data = {{ nodes: nodes, edges: edges }};
                
                // Initialize with hierarchical layout for proper layer-like positioning
                // Direction 'UD' = top-to-bottom flow (rotated 90° clockwise from LR)
                const options = {{
                    nodes: {{
                        shape: 'dot',
                        size: 18,
                        font: {{ size: 11 }},
                        borderWidth: 2
                    }},
                    edges: {{
                        arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
                        smooth: {{ type: 'curvedCW', roundness: 0.1 }}
                    }},
                    layout: {{
                        hierarchical: {{
                            enabled: true,
                            direction: 'UD',
                            sortMethod: 'directed',
                            levelSeparation: 120,
                            nodeSpacing: 100,
                            treeSpacing: 120
                        }}
                    }},
                    physics: {{
                        enabled: false
                    }},
                    interaction: {{
                        hover: true,
                        tooltipDelay: 50,
                        multiselect: true,
                        dragNodes: true
                    }}
                }};
                
                const network = new vis.Network(container, data, options);
                
                // After initial layout, disable hierarchical for free movement
                setTimeout(function() {{
                    network.setOptions({{ layout: {{ hierarchical: false }} }});
                    network.fit({{ animation: true }});
                }}, 200);
                
                // Register with global toggle (store original data for filtering)
                window.allNetworks[{threshold}] = {{
                    network: network,
                    nodes: nodes,
                    edges: edges,
                    allNodes: allNodes,
                    allEdges: allEdges,
                    conservedEdgeIds: conservedEdgeIds,
                    conservedNodeIds: conservedNodeIds,
                    deadEndNodeIds: deadEndNodeIds
                }};
            }})();
        </script>
'''


def _generate_edge_matrices_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                     nickname_map: Dict[str, str]) -> str:
    """Generate edge presence matrices section with dual toggle (by threshold and by dataset)."""
    html_parts = []
    nicknames = [nickname_map[d] for d in dataset_names]
    nicknames_json = json.dumps(nicknames)
    thresholds_json = json.dumps(thresholds)
    
    html_parts.append(f"""
        <div id="edge-matrices" class="section">
            <div class="section-header">🔗 Edge Presence Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Edge presence across datasets. ✔️ = present with weight, ❌ = absent.
                </p>
                
                <!-- Toggle Mode Selector -->
                <div style="margin-bottom: 15px;">
                    <span style="font-weight: 600; margin-right: 10px;">View by:</span>
                    <button class="tab-btn active" id="edge_mode_threshold" onclick="switchEdgeMode('threshold')">Threshold</button>
                    <button class="tab-btn" id="edge_mode_dataset" onclick="switchEdgeMode('dataset')">Dataset</button>
                </div>
                
                <!-- By Threshold View -->
                <div id="edge_by_threshold" class="tabs">
                    <div class="tab-buttons">
""")
    
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<button class="tab-btn {active}" onclick="showEdgeTab({t})">t = {t}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="edge_tab_{threshold}" class="tab-content {active}">')
        html_parts.append(_generate_presence_table(analyzer.get_aligned_data(threshold), dataset_names, nickname_map, threshold))
        html_parts.append('</div>')
    
    html_parts.append('</div>')  # Close edge_by_threshold
    
    # By Dataset View
    html_parts.append(f'''
                <!-- By Dataset View -->
                <div id="edge_by_dataset" class="tabs" style="display: none;">
                    <div class="tab-buttons">
''')
    
    for i, d in enumerate(dataset_names):
        active = 'active' if i == 0 else ''
        nick = nickname_map[d]
        html_parts.append(f'<button class="tab-btn {active}" onclick="showEdgeDatasetTab(\'{nick}\')">{nick}</button>')
    html_parts.append('</div>')
    
    # Generate dataset-centric tables (showing all thresholds for one dataset)
    for i, d in enumerate(dataset_names):
        nick = nickname_map[d]
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="edge_dataset_tab_{nick}" class="tab-content {active}">')
        html_parts.append(_generate_edge_dataset_table(analyzer, d, thresholds, nickname_map))
        html_parts.append('</div>')
    
    html_parts.append(f"""
                </div>
                
                <script>
                    function switchEdgeMode(mode) {{
                        document.getElementById('edge_mode_threshold').classList.toggle('active', mode === 'threshold');
                        document.getElementById('edge_mode_dataset').classList.toggle('active', mode === 'dataset');
                        document.getElementById('edge_by_threshold').style.display = mode === 'threshold' ? 'block' : 'none';
                        document.getElementById('edge_by_dataset').style.display = mode === 'dataset' ? 'block' : 'none';
                    }}
                    
                    function showEdgeTab(threshold) {{
                        document.querySelectorAll('#edge_by_threshold .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#edge_by_threshold .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('edge_tab_' + threshold).classList.add('active');
                        event.target.classList.add('active');
                    }}
                    
                    function showEdgeDatasetTab(dataset) {{
                        document.querySelectorAll('#edge_by_dataset .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#edge_by_dataset .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('edge_dataset_tab_' + dataset).classList.add('active');
                        event.target.classList.add('active');
                    }}
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_edge_dataset_table(analyzer, dataset: str, thresholds: List[int], 
                                  nickname_map: Dict[str, str]) -> str:
    """Generate edge presence table for a single dataset across all thresholds."""
    nick = nickname_map[dataset]
    
    # Collect all edges and their presence/weight across thresholds
    all_edges = set()
    edge_data = {}  # edge_key -> {threshold: weight}
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        if aligned.empty or dataset not in aligned.columns:
            continue
        for edge_key, row in aligned.iterrows():
            weight = row.get(dataset, 0)
            all_edges.add(edge_key)
            if edge_key not in edge_data:
                edge_data[edge_key] = {}
            edge_data[edge_key][threshold] = weight
    
    if not edge_data:
        return '<p>No data available.</p>'
    
    # Sort edges by lowest threshold weight
    sorted_edges = sorted(
        edge_data.items(), 
        key=lambda x: (-x[1].get(thresholds[0], 0), str(x[0]))
    )[:50]
    
    html = [f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Dataset: {nick} (all thresholds)</div>']
    html.append('<div style="overflow-x: auto;"><table><thead><tr><th>Edge</th>')
    for t in thresholds:
        html.append(f'<th>t={t}</th>')
    html.append('</tr></thead><tbody>')
    
    for edge_key, weights in sorted_edges:
        html.append(f'<tr><td><strong>{edge_key}</strong></td>')
        for t in thresholds:
            w = weights.get(t, 0)
            if w > 0:
                html.append(f'<td><span class="presence-check">✔️</span> {int(w)}</td>')
            else:
                html.append('<td><span class="presence-cross">❌</span></td>')
        html.append('</tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _generate_path_matrices_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                     nickname_map: Dict[str, str]) -> str:
    """Generate path presence matrices section with dual toggle (by threshold and by dataset)."""
    html_parts = []
    nicknames = [nickname_map[d] for d in dataset_names]
    
    html_parts.append(f"""
        <div id="path-matrices" class="section">
            <div class="section-header">🛤️ Path Presence Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Path presence across datasets. Weight = minimum edge weight along path.<br>
                    <em>Hop weights shown as <code>-w₁-w₂-</code> with <strong>min weight bolded</strong>.</em>
                </p>
                
                <!-- Toggle Mode Selector -->
                <div style="margin-bottom: 15px;">
                    <span style="font-weight: 600; margin-right: 10px;">View by:</span>
                    <button class="tab-btn active" id="path_mode_threshold" onclick="switchPathMode('threshold')">Threshold</button>
                    <button class="tab-btn" id="path_mode_dataset" onclick="switchPathMode('dataset')">Dataset</button>
                </div>
                
                <!-- By Threshold View -->
                <div id="path_by_threshold" class="tabs">
                    <div class="tab-buttons">
""")
    
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<button class="tab-btn {active}" onclick="showPathTab({t})">t = {t}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="path_tab_{threshold}" class="tab-content {active}">')
        path_data = analyzer._get_path_data_for_threshold(threshold)
        if path_data is not None and not path_data.empty:
            html_parts.append(_generate_path_presence_table(analyzer, path_data, dataset_names, nickname_map, threshold))
        else:
            html_parts.append('<p>No path data available.</p>')
        html_parts.append('</div>')
    
    html_parts.append('</div>')  # Close path_by_threshold
    
    # By Dataset View
    html_parts.append(f'''
                <!-- By Dataset View -->
                <div id="path_by_dataset" class="tabs" style="display: none;">
                    <div class="tab-buttons">
''')
    
    for i, d in enumerate(dataset_names):
        active = 'active' if i == 0 else ''
        nick = nickname_map[d]
        html_parts.append(f'<button class="tab-btn {active}" onclick="showPathDatasetTab(\'{nick}\')">{nick}</button>')
    html_parts.append('</div>')
    
    # Generate dataset-centric tables (showing all thresholds for one dataset)
    for i, d in enumerate(dataset_names):
        nick = nickname_map[d]
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="path_dataset_tab_{nick}" class="tab-content {active}">')
        html_parts.append(_generate_path_dataset_table(analyzer, d, thresholds, nickname_map))
        html_parts.append('</div>')
    
    html_parts.append(f"""
                </div>
                
                <script>
                    function switchPathMode(mode) {{
                        document.getElementById('path_mode_threshold').classList.toggle('active', mode === 'threshold');
                        document.getElementById('path_mode_dataset').classList.toggle('active', mode === 'dataset');
                        document.getElementById('path_by_threshold').style.display = mode === 'threshold' ? 'block' : 'none';
                        document.getElementById('path_by_dataset').style.display = mode === 'dataset' ? 'block' : 'none';
                    }}
                    
                    function showPathTab(threshold) {{
                        document.querySelectorAll('#path_by_threshold .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#path_by_threshold .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('path_tab_' + threshold).classList.add('active');
                        event.target.classList.add('active');
                    }}
                    
                    function showPathDatasetTab(dataset) {{
                        document.querySelectorAll('#path_by_dataset .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#path_by_dataset .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('path_dataset_tab_' + dataset).classList.add('active');
                        event.target.classList.add('active');
                    }}
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_path_presence_table(analyzer, data: pd.DataFrame, dataset_names: List[str], 
                                   nickname_map: Dict[str, str], threshold: int = None) -> str:
    """Generate path presence table with hop weights shown as -w1-w2- with min bolded."""
    if data is None or data.empty:
        return '<p>No data available.</p>'
    
    available = [d for d in dataset_names if d in data.columns]
    if not available:
        return '<p>No datasets available.</p>'
    
    # Get path hop weights from analyzer
    path_hop_weights = analyzer._get_path_hop_weights_for_threshold(threshold) if hasattr(analyzer, '_get_path_hop_weights_for_threshold') else {}
    
    # Add threshold caption if provided
    threshold_caption = f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Threshold = {threshold}</div>' if threshold is not None else ''
    
    html = [f'{threshold_caption}<div style="overflow-x: auto;"><table><thead><tr><th>Item</th>']
    for d in available:
        html.append(f'<th>{nickname_map[d]}</th>')
    html.append('<th>Conservation</th></tr></thead><tbody>')
    
    # Sort by conservation (count of datasets present), then by total weight
    data_copy = data.copy()
    data_copy['_conservation'] = (data_copy[available] > 0).sum(axis=1)
    data_copy['_total'] = data_copy[available].sum(axis=1)
    data_copy = data_copy.sort_values(['_conservation', '_total'], ascending=[False, False])
    top = data_copy.head(min(50, len(data_copy)))
    
    for key, row in top.iterrows():
        count = sum(1 for d in available if row.get(d, 0) > 0)
        badge = 'badge-success' if count == len(available) else 'badge-warning' if count > 1 else 'badge-danger'
        
        html.append(f'<tr><td><strong>{key}</strong></td>')
        for d in available:
            w = row.get(d, 0)
            safe_name = analyzer.parameters._sanitize_name(d)
            if w > 0:
                # Get hop weights if available
                hop_weights_str = ''
                if key in path_hop_weights and safe_name in path_hop_weights[key]:
                    hop_weights = path_hop_weights[key][safe_name]
                    if hop_weights:
                        min_w = min(hop_weights)
                        # Format with bolded minimum
                        formatted = []
                        for hw in hop_weights:
                            if hw == min_w:
                                formatted.append(f'<strong>{int(hw)}</strong>')
                            else:
                                formatted.append(str(int(hw)))
                        hop_weights_str = f'<br><span style="font-size: 0.8em; color: #666;">-{"-".join(formatted)}-</span>'
                
                html.append(f'<td><span class="presence-check">✔️</span> {int(w)}{hop_weights_str}</td>')
            else:
                html.append('<td><span class="presence-cross">❌</span></td>')
        html.append(f'<td><span class="badge {badge}">{count}/{len(available)}</span></td></tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _generate_path_dataset_table(analyzer, dataset: str, thresholds: List[int], 
                                  nickname_map: Dict[str, str]) -> str:
    """Generate path presence table for a single dataset across all thresholds."""
    nick = nickname_map[dataset]
    safe_name = analyzer.parameters._sanitize_name(dataset)
    
    # Collect all paths and their presence/weight across thresholds
    all_paths = set()
    path_data = {}  # path_key -> {threshold: (weight, hop_weights)}
    
    for threshold in thresholds:
        pdata = analyzer._get_path_data_for_threshold(threshold)
        if pdata is None or pdata.empty or dataset not in pdata.columns:
            continue
        
        # Get hop weights for this threshold
        hop_weights_dict = analyzer._get_path_hop_weights_for_threshold(threshold) if hasattr(analyzer, '_get_path_hop_weights_for_threshold') else {}
        
        for path_key, row in pdata.iterrows():
            weight = row.get(dataset, 0)
            all_paths.add(path_key)
            if path_key not in path_data:
                path_data[path_key] = {}
            
            hop_weights = hop_weights_dict.get(path_key, {}).get(safe_name, []) if hop_weights_dict else []
            path_data[path_key][threshold] = (weight, hop_weights)
    
    if not path_data:
        return '<p>No data available.</p>'
    
    # Sort paths by lowest threshold weight
    sorted_paths = sorted(
        path_data.items(), 
        key=lambda x: (-x[1].get(thresholds[0], (0, []))[0], str(x[0]))
    )[:50]
    
    html = [f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Dataset: {nick} (all thresholds)</div>']
    html.append('<div style="overflow-x: auto;"><table><thead><tr><th>Path</th>')
    for t in thresholds:
        html.append(f'<th>t={t}</th>')
    html.append('</tr></thead><tbody>')
    
    for path_key, weights in sorted_paths:
        html.append(f'<tr><td><strong>{path_key}</strong></td>')
        for t in thresholds:
            w, hop_weights = weights.get(t, (0, []))
            if w > 0:
                hop_weights_str = ''
                if hop_weights:
                    min_w = min(hop_weights)
                    formatted = []
                    for hw in hop_weights:
                        if hw == min_w:
                            formatted.append(f'<strong>{int(hw)}</strong>')
                        else:
                            formatted.append(str(int(hw)))
                    hop_weights_str = f'<br><span style="font-size: 0.8em; color: #666;">-{"-".join(formatted)}-</span>'
                html.append(f'<td><span class="presence-check">✔️</span> {int(w)}{hop_weights_str}</td>')
            else:
                html.append('<td><span class="presence-cross">❌</span></td>')
        html.append('</tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _generate_presence_table(data: pd.DataFrame, dataset_names: List[str], 
                              nickname_map: Dict[str, str], threshold: int = None) -> str:
    """Generate a presence matrix table with optional threshold indication."""
    if data is None or data.empty:
        return '<p>No data available.</p>'
    
    available = [d for d in dataset_names if d in data.columns]
    if not available:
        return '<p>No datasets available.</p>'
    
    # Add threshold caption if provided
    threshold_caption = f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Threshold = {threshold}</div>' if threshold is not None else ''
    
    html = [f'{threshold_caption}<div style="overflow-x: auto;"><table><thead><tr><th>Item</th>']
    for d in available:
        html.append(f'<th>{nickname_map[d]}</th>')
    html.append('<th>Conservation</th></tr></thead><tbody>')
    
    # Sort by conservation (count of datasets present), then by total weight
    data_copy = data.copy()
    data_copy['_conservation'] = (data_copy[available] > 0).sum(axis=1)
    data_copy['_total'] = data_copy[available].sum(axis=1)
    # Sort by conservation descending, then by total descending
    data_copy = data_copy.sort_values(['_conservation', '_total'], ascending=[False, False])
    top = data_copy.head(min(50, len(data_copy)))
    
    for key, row in top.iterrows():
        count = sum(1 for d in available if row.get(d, 0) > 0)
        badge = 'badge-success' if count == len(available) else 'badge-warning' if count > 1 else 'badge-danger'
        
        html.append(f'<tr><td><strong>{key}</strong></td>')
        for d in available:
            w = row.get(d, 0)
            if w > 0:
                html.append(f'<td><span class="presence-check">✔️</span> {int(w)}</td>')
            else:
                html.append('<td><span class="presence-cross">❌</span></td>')
        html.append(f'<td><span class="badge {badge}">{count}/{len(available)}</span></td></tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _generate_conservation_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                    key_findings: Dict, nickname_map: Dict[str, str]) -> str:
    """Generate conservation analysis section with distribution by dataset count."""
    num_datasets = len(dataset_names)
    html_parts = []
    html_parts.append(f"""
        <div id="conservation" class="section">
            <div class="section-header">🏆 Conservation Analysis</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Edge and path conservation across all {num_datasets} datasets. Shows distribution by how many datasets each edge/path appears in.
                </p>
                <div class="grid grid-2">
""")
    
    # Color palette for conservation levels (from conserved to unique)
    # Using a gradient from green (conserved in all) to gray (unique)
    conservation_colors = [
        '#22c55e',  # All datasets (green)
        '#84cc16',  # N-1 (lime)
        '#eab308',  # N-2 (yellow)
        '#f59e0b',  # N-3 (orange)
        '#f97316',  # N-4 (orange-red)
        '#ef4444',  # Lower (red)
        '#94a3b8',  # Unique (gray)
    ]
    
    for threshold in thresholds:
        # Get aligned data to compute distribution
        aligned = analyzer.get_aligned_data(threshold)
        available = [d for d in dataset_names if d in aligned.columns]
        n = len(available)
        
        if aligned.empty or n == 0:
            continue
        
        # Compute edge distribution: count how many datasets each edge appears in
        edge_counts = {}  # {count: number_of_edges}
        for edge_key, row in aligned.iterrows():
            count = sum(1 for d in available if row.get(d, 0) > 0)
            if count > 0:
                edge_counts[count] = edge_counts.get(count, 0) + 1
        
        # Get path data if available
        path_counts = {}  # {count: number_of_paths}
        try:
            # Try to get path data from analyzer
            if hasattr(analyzer, '_get_path_data_for_threshold'):
                path_data = analyzer._get_path_data_for_threshold(threshold)
            elif hasattr(analyzer, 'get_path_data'):
                path_data = analyzer.get_path_data(threshold)
            else:
                path_data = None
            
            if path_data is not None and not path_data.empty:
                for _, row in path_data.iterrows():
                    count = sum(1 for d in available if d in row.index and row.get(d, 0) > 0)
                    if count > 0:
                        path_counts[count] = path_counts.get(count, 0) + 1
        except:
            pass
        
        # Build data for pie charts
        edge_values = []
        edge_labels = []
        edge_colors = []
        
        for count in range(n, 0, -1):
            if count in edge_counts and edge_counts[count] > 0:
                edge_values.append(edge_counts[count])
                if count == n:
                    edge_labels.append(f'All {n} datasets')
                elif count == 1:
                    edge_labels.append('Unique (1)')
                else:
                    edge_labels.append(f'In {count} datasets')
                # Assign color based on position from top
                color_idx = min(n - count, len(conservation_colors) - 1)
                edge_colors.append(conservation_colors[color_idx])
        
        path_values = []
        path_labels = []
        path_colors = []
        
        for count in range(n, 0, -1):
            if count in path_counts and path_counts[count] > 0:
                path_values.append(path_counts[count])
                if count == n:
                    path_labels.append(f'All {n} datasets')
                elif count == 1:
                    path_labels.append('Unique (1)')
                else:
                    path_labels.append(f'In {count} datasets')
                color_idx = min(n - count, len(conservation_colors) - 1)
                path_colors.append(conservation_colors[color_idx])
        
        # Summary stats
        kf = key_findings.get(threshold, {})
        te, ce = kf.get('total_edges', 0), kf.get('common_edges', 0)
        tp, cp = kf.get('total_paths', 0), kf.get('common_paths', 0)
        er = (ce / te * 100) if te > 0 else 0
        pr = (cp / tp * 100) if tp > 0 else 0
        
        # Convert to JSON for JavaScript
        edge_values_json = json.dumps(edge_values)
        edge_labels_json = json.dumps(edge_labels)
        edge_colors_json = json.dumps(edge_colors)
        path_values_json = json.dumps(path_values)
        path_labels_json = json.dumps(path_labels)
        path_colors_json = json.dumps(path_colors)
        
        html_parts.append(f'''
            <div class="card">
                <h3>Conservation at Threshold = {threshold}</h3>
                <div class="grid grid-2">
                    <div id="cons_edge_{threshold}" style="height: 260px;"></div>
                    <div id="cons_path_{threshold}" style="height: 260px;"></div>
                </div>
                <div style="text-align: center; color: var(--secondary-color); font-size: 0.85rem; margin-top: 10px;">
                    Edges: {ce}/{te} conserved ({er:.1f}%) | Paths: {cp}/{tp} conserved ({pr:.1f}%)
                </div>
            </div>
            <script>
                (function() {{
                    const edgeValues = {edge_values_json};
                    const edgeLabels = {edge_labels_json};
                    const edgeColors = {edge_colors_json};
                    const pathValues = {path_values_json};
                    const pathLabels = {path_labels_json};
                    const pathColors = {path_colors_json};
                    
                    if (edgeValues.length > 0) {{
                        Plotly.newPlot('cons_edge_{threshold}', [{{
                            values: edgeValues, 
                            labels: edgeLabels,
                            type: 'pie', 
                            hole: 0.35, 
                            marker: {{ colors: edgeColors }},
                            textinfo: 'percent',
                            textposition: 'inside',
                            hoverinfo: 'label+value+percent'
                        }}], {{ 
                            title: {{ text: 'Edges', font: {{ size: 13 }} }}, 
                            showlegend: true,
                            legend: {{ font: {{ size: 9 }}, x: 0, y: -0.15, orientation: 'h' }},
                            margin: {{ t: 40, b: 50, l: 10, r: 10 }} 
                        }}, {{responsive: true}});
                    }} else {{
                        document.getElementById('cons_edge_{threshold}').innerHTML = '<p style="text-align:center;color:#999;">No edge data</p>';
                    }}
                    
                    if (pathValues.length > 0) {{
                        Plotly.newPlot('cons_path_{threshold}', [{{
                            values: pathValues, 
                            labels: pathLabels,
                            type: 'pie', 
                            hole: 0.35, 
                            marker: {{ colors: pathColors }},
                            textinfo: 'percent',
                            textposition: 'inside',
                            hoverinfo: 'label+value+percent'
                        }}], {{ 
                            title: {{ text: 'Paths', font: {{ size: 13 }} }}, 
                            showlegend: true,
                            legend: {{ font: {{ size: 9 }}, x: 0, y: -0.15, orientation: 'h' }},
                            margin: {{ t: 40, b: 50, l: 10, r: 10 }} 
                        }}, {{responsive: true}});
                    }} else {{
                        document.getElementById('cons_path_{threshold}').innerHTML = '<p style="text-align:center;color:#999;">No path data</p>';
                    }}
                }})();
            </script>
''')
    
    html_parts.append('</div></div></div>')
    return ''.join(html_parts)


def _generate_overlap_matrices_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                        nickname_map: Dict[str, str]) -> str:
    """Generate dataset overlap matrices section.
    
    Shows asymmetric N×N matrices where cell (i,j) = 
    number of edges/paths in dataset i that are also found in dataset j.
    Diagonal shows total count for each dataset.
    """
    html_parts = []
    nicknames = [nickname_map[d] for d in dataset_names]
    thresholds_json = json.dumps(thresholds)
    
    html_parts.append(f"""
        <div id="overlap-matrices" class="section">
            <div class="section-header">🔀 Dataset Overlap Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Asymmetric overlap matrices. Cell (row, col) shows how many edges/paths from the <strong>row</strong> dataset 
                    are also found in the <strong>column</strong> dataset. Diagonal = total count per dataset.
                </p>
                <div class="tabs">
                    <div class="tab-buttons">
""")
    
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<button class="tab-btn {active}" onclick="showOverlapTab({t})">t = {t}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        aligned = analyzer.get_aligned_data(threshold)
        available = [d for d in dataset_names if d in aligned.columns]
        n = len(available)
        
        if aligned.empty or n == 0:
            active = 'active' if i == 0 else ''
            html_parts.append(f'<div id="overlap_tab_{threshold}" class="tab-content {active}"><p>No data at this threshold.</p></div>')
            continue
        
        # Compute edge overlap matrix (asymmetric)
        edge_overlap = [[0 for _ in range(n)] for _ in range(n)]
        for i1, d1 in enumerate(available):
            edges_in_d1 = set(aligned.index[aligned[d1] > 0])
            edge_overlap[i1][i1] = len(edges_in_d1)  # Diagonal = total
            for i2, d2 in enumerate(available):
                if i1 != i2:
                    edges_in_d2 = set(aligned.index[aligned[d2] > 0])
                    # Edges in d1 that are also in d2
                    edge_overlap[i1][i2] = len(edges_in_d1 & edges_in_d2)
        
        # Compute path overlap matrix (asymmetric)
        path_overlap = [[0 for _ in range(n)] for _ in range(n)]
        try:
            path_data = analyzer._get_path_data_for_threshold(threshold)
            if path_data is not None and not path_data.empty:
                for i1, d1 in enumerate(available):
                    paths_in_d1 = set(path_data.index[path_data[d1] > 0]) if d1 in path_data.columns else set()
                    path_overlap[i1][i1] = len(paths_in_d1)
                    for i2, d2 in enumerate(available):
                        if i1 != i2:
                            paths_in_d2 = set(path_data.index[path_data[d2] > 0]) if d2 in path_data.columns else set()
                            path_overlap[i1][i2] = len(paths_in_d1 & paths_in_d2)
        except Exception as e:
            pass  # Path data may not be available
        
        # Calculate chart size - make matrices larger (~1/3 page width) and square
        # For a 1600px max-width container with padding, aim for ~450px per matrix
        chart_size = 450  # Fixed size for square matrices
        
        labels = [nickname_map[d] for d in available]
        edge_overlap_json = json.dumps(edge_overlap)
        path_overlap_json = json.dumps(path_overlap)
        labels_json = json.dumps(labels)
        
        active = 'active' if i == 0 else ''
        html_parts.append(f'''
            <div id="overlap_tab_{threshold}" class="tab-content {active}">
                <div class="card">
                    <h3>Dataset Overlap at Threshold = {threshold}</h3>
                    <div style="margin-bottom: 15px; text-align: center;">
                        <label style="margin-right: 20px; cursor: pointer;">
                            <input type="radio" name="overlap_mode_{threshold}" value="count" checked 
                                   onclick="updateOverlapMode_{threshold}('count')"> Show Count
                        </label>
                        <label style="cursor: pointer;">
                            <input type="radio" name="overlap_mode_{threshold}" value="proportion" 
                                   onclick="updateOverlapMode_{threshold}('proportion')"> Show Proportion
                        </label>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 30px; justify-content: center;">
                        <div style="width: {chart_size}px;">
                            <h5 style="text-align: center; margin-bottom: 8px; color: var(--secondary-color);">Edge Overlap</h5>
                            <div id="edge_overlap_{threshold}" style="width: {chart_size}px; height: {chart_size}px;"></div>
                        </div>
                        <div style="width: {chart_size}px;">
                            <h5 style="text-align: center; margin-bottom: 8px; color: var(--secondary-color);">Path Overlap</h5>
                            <div id="path_overlap_{threshold}" style="width: {chart_size}px; height: {chart_size}px;"></div>
                        </div>
                    </div>
                    <p style="font-size: 0.8em; color: #64748b; margin-top: 10px; text-align: center;">
                        Read as: edges/paths from <strong>row</strong> dataset found in <strong>column</strong> dataset.
                        Diagonal = total in that dataset.
                    </p>
                </div>
            </div>
            <script>
                (function() {{
                    const labels_{threshold} = {labels_json};
                    const edgeOverlap_{threshold} = {edge_overlap_json};
                    const pathOverlap_{threshold} = {path_overlap_json};
                    
                    // Compute proportion matrices (row-normalized: what % of row's edges/paths are in col)
                    const edgeProportion_{threshold} = edgeOverlap_{threshold}.map((row, i) => 
                        row.map((val, j) => {{
                            const diag = edgeOverlap_{threshold}[i][i];
                            return diag > 0 ? val / diag : 0;
                        }})
                    );
                    const pathProportion_{threshold} = pathOverlap_{threshold}.map((row, i) => 
                        row.map((val, j) => {{
                            const diag = pathOverlap_{threshold}[i][i];
                            return diag > 0 ? val / diag : 0;
                        }})
                    );
                    
                    // Create text annotations for count mode
                    const edgeTextCount_{threshold} = edgeOverlap_{threshold}.map((row, i) => 
                        row.map((val, j) => {{
                            const diag = edgeOverlap_{threshold}[i][i];
                            const pct = diag > 0 ? (val / diag * 100).toFixed(0) : 0;
                            return i === j ? String(val) : val + ' (' + pct + '%)';
                        }})
                    );
                    const pathTextCount_{threshold} = pathOverlap_{threshold}.map((row, i) => 
                        row.map((val, j) => {{
                            const diag = pathOverlap_{threshold}[i][i];
                            const pct = diag > 0 ? (val / diag * 100).toFixed(0) : 0;
                            return i === j ? String(val) : val + ' (' + pct + '%)';
                        }})
                    );
                    
                    // Create text annotations for proportion mode
                    const edgeTextProp_{threshold} = edgeProportion_{threshold}.map((row, i) => 
                        row.map((val, j) => (val * 100).toFixed(1) + '%')
                    );
                    const pathTextProp_{threshold} = pathProportion_{threshold}.map((row, i) => 
                        row.map((val, j) => (val * 100).toFixed(1) + '%')
                    );
                    
                    // Square matrix layout with fixed aspect ratio
                    const baseLayout = {{
                        xaxis: {{ tickangle: -45, side: 'bottom', tickfont: {{size: 11}}, constrain: 'domain' }},
                        yaxis: {{ autorange: 'reversed', tickfont: {{size: 11}}, scaleanchor: 'x', scaleratio: 1 }},
                        margin: {{ l: 100, r: 30, t: 30, b: 100 }},
                        width: {chart_size},
                        height: {chart_size}
                    }};
                    
                    // Store current mode
                    window.overlapMode_{threshold} = 'count';
                    
                    // Update function for toggling modes
                    window.updateOverlapMode_{threshold} = function(mode) {{
                        window.overlapMode_{threshold} = mode;
                        
                        if (mode === 'count') {{
                            // Count mode
                            Plotly.react('edge_overlap_{threshold}', [{{
                                z: edgeOverlap_{threshold},
                                x: labels_{threshold},
                                y: labels_{threshold},
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#93c5fd'], [1, '#2563eb']],
                                text: edgeTextCount_{threshold},
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z}} edges<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Count', len: 0.5 }}
                            }}], baseLayout);
                            
                            Plotly.react('path_overlap_{threshold}', [{{
                                z: pathOverlap_{threshold},
                                x: labels_{threshold},
                                y: labels_{threshold},
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#c4b5fd'], [1, '#8b5cf6']],
                                text: pathTextCount_{threshold},
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z}} paths<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Count', len: 0.5 }}
                            }}], baseLayout);
                        }} else {{
                            // Proportion mode (row-normalized)
                            Plotly.react('edge_overlap_{threshold}', [{{
                                z: edgeProportion_{threshold},
                                x: labels_{threshold},
                                y: labels_{threshold},
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#93c5fd'], [1, '#2563eb']],
                                text: edgeTextProp_{threshold},
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z:.1%}} of row edges<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Proportion', len: 0.5, tickformat: '.0%' }},
                                zmin: 0, zmax: 1
                            }}], baseLayout);
                            
                            Plotly.react('path_overlap_{threshold}', [{{
                                z: pathProportion_{threshold},
                                x: labels_{threshold},
                                y: labels_{threshold},
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#c4b5fd'], [1, '#8b5cf6']],
                                text: pathTextProp_{threshold},
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z:.1%}} of row paths<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Proportion', len: 0.5, tickformat: '.0%' }},
                                zmin: 0, zmax: 1
                            }}], baseLayout);
                        }}
                    }};
                    
                    // Initial plot in count mode
                    Plotly.newPlot('edge_overlap_{threshold}', [{{
                        z: edgeOverlap_{threshold},
                        x: labels_{threshold},
                        y: labels_{threshold},
                        type: 'heatmap',
                        colorscale: [[0, '#f8fafc'], [0.5, '#93c5fd'], [1, '#2563eb']],
                        text: edgeTextCount_{threshold},
                        texttemplate: '%{{text}}',
                        textfont: {{ size: 10 }},
                        hovertemplate: '%{{y}} in %{{x}}: %{{z}} edges<extra></extra>',
                        showscale: true,
                        colorbar: {{ title: 'Count', len: 0.5 }}
                    }}], baseLayout, {{responsive: false}});
                    
                    Plotly.newPlot('path_overlap_{threshold}', [{{
                        z: pathOverlap_{threshold},
                        x: labels_{threshold},
                        y: labels_{threshold},
                        type: 'heatmap',
                        colorscale: [[0, '#f8fafc'], [0.5, '#c4b5fd'], [1, '#8b5cf6']],
                        text: pathTextCount_{threshold},
                        texttemplate: '%{{text}}',
                        textfont: {{ size: 10 }},
                        hovertemplate: '%{{y}} in %{{x}}: %{{z}} paths<extra></extra>',
                        showscale: true,
                        colorbar: {{ title: 'Count', len: 0.5 }}
                    }}], baseLayout, {{responsive: false}});
                }})();
            </script>
''')
    
    html_parts.append(f"""
                </div>
                <script>
                    function showOverlapTab(threshold) {{
                        document.querySelectorAll('#overlap-matrices .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#overlap-matrices .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('overlap_tab_' + threshold).classList.add('active');
                        event.target.classList.add('active');
                    }}
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_statistics_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                  nickname_map: Dict[str, str]) -> str:
    """Generate statistics section."""
    html_parts = []
    html_parts.append("""
        <div id="statistics" class="section">
            <div class="section-header">📉 Statistics</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Detailed statistics per dataset and threshold.
                </p>
""")
    
    html_parts.append('<div class="tabs"><div class="tab-buttons">')
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<button class="tab-btn {active}" onclick="showStatsTab({t})">t = {t}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="stats_tab_{threshold}" class="tab-content {active}">')
        html_parts.append(_generate_stats_table(analyzer, dataset_names, threshold, nickname_map))
        html_parts.append('</div>')
    
    html_parts.append("""
                </div>
                <script>
                    function showStatsTab(threshold) {
                        document.querySelectorAll('#statistics .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#statistics .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('stats_tab_' + threshold).classList.add('active');
                        event.target.classList.add('active');
                    }
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_stats_table(analyzer, dataset_names: List[str], threshold: int,
                           nickname_map: Dict[str, str]) -> str:
    """Generate statistics tables with threshold indication."""
    aligned = analyzer.get_aligned_data(threshold)
    
    if aligned.empty:
        return '<p>No data available.</p>'
    
    available = [d for d in dataset_names if d in aligned.columns]
    if not available:
        return '<p>No datasets available.</p>'
    
    html = [f'''<div class="card"><h3>Per-Dataset Statistics <span style="color: var(--primary-color);">(Threshold = {threshold})</span></h3>
        <table><thead><tr><th>Dataset</th><th>Edge Count</th><th>Total Weight</th><th>Mean Weight</th><th>Max Weight</th></tr></thead><tbody>''']
    
    for d in available:
        col = aligned[d]
        ec = int((col > 0).sum())
        tw = int(col.sum())
        mw = float(col[col > 0].mean()) if (col > 0).any() else 0
        mx = int(col.max()) if len(col) > 0 else 0
        html.append(f'<tr><td>{nickname_map[d]}</td><td>{ec}</td><td>{tw}</td><td>{mw:.2f}</td><td>{mx}</td></tr>')
    
    html.append('</tbody></table></div>')
    
    # Pairwise similarities
    html.append(f'''<div class="card"><h3>Pairwise Similarities <span style="color: var(--primary-color);">(Threshold = {threshold})</span></h3>
        <table><thead><tr><th>Dataset 1</th><th>Dataset 2</th><th>Jaccard</th><th>Cosine</th><th>Common</th></tr></thead><tbody>''')
    
    for i, d1 in enumerate(available):
        for d2 in available[i+1:]:
            c1, c2 = aligned[d1], aligned[d2]
            s1, s2 = set(aligned.index[c1 > 0]), set(aligned.index[c2 > 0])
            inter, union = len(s1 & s2), len(s1 | s2)
            jac = inter / union if union > 0 else 0
            w1, w2 = c1.values, c2.values
            cos = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2)) if np.linalg.norm(w1) > 0 and np.linalg.norm(w2) > 0 else 0
            html.append(f'<tr><td>{nickname_map[d1]}</td><td>{nickname_map[d2]}</td><td>{jac:.3f}</td><td>{cos:.3f}</td><td>{inter}</td></tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _generate_footer() -> str:
    """Generate footer."""
    return """
        <button class="print-btn" onclick="window.print()">🖨️ Print Report</button>
    </div>
</body>
</html>
"""

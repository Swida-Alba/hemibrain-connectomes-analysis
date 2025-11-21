# Documentation Index

Welcome to the Hemibrain Connectomes Analysis documentation!

## 🆕 Recent Updates (November 2025)

- **[Visualization Update Summary](./VISUALIZATION_UPDATE.md)** - Summary of all visualization-related updates and deprecations.
- **[VisualizePath Updates Nov 2025](./visualizations/VisualizePath_Updates_Nov2025.md)** - Detailed guide on connection matrix support and new features.
- **[VisualizePath Standalone Reorganization](./VISPATH_STANDALONE_REORGANIZATION.md)** - vispath now fully standalone, no statvis dependency
- **[Import Strategy Update](./IMPORT_STRATEGY_UPDATE.md)** - Scripts/examples now use vispath-subproject
- **[vispath-subproject README](../vispath-subproject/README.md)** - Corrected data format documentation
- **[vispath-subproject Installation](../vispath-subproject/INSTALLATION.md)** - Added data format examples

## 📚 Documentation Structure

Documentation is organized into three main categories:

### 🎨 [Visualizations](./visualizations/)
Comprehensive guides for all visualization types:
- **[Heatmap](./visualizations/Heatmap_Guide.md)**: Interactive connection matrices
- **[Network](./visualizations/Network_Guide.md)**: Graph-based connectivity visualization
- **[Sankey](./visualizations/Sankey_Guide.md)**: Flow diagrams showing connection magnitude
- **[3D Skeleton](./visualizations/3D_Skeleton_Guide.md)**: Three-dimensional neuron morphology

### 🔧 [Core Features](./core-features/)
Essential functionality documentation:
- **Path Finding**: Multi-hop connection discovery
- **Custom Groups**: Flexible neuron grouping for custom analysis (NEW)
- **Cache System**: High-performance local storage (10-100x speedup)
- **Parallel Processing**: Multi-core acceleration (4-14x speedup)
- **Filtering**: Connection and neuron filtering options

### ⚙️ [Technical](./technical/)
Advanced technical documentation:
- **Performance Optimization**: Algorithm improvements and profiling
- **Data Formats**: Input/output specifications
- **Backend Architecture**: System design and implementation
- **Debugging**: Troubleshooting and development guides

---

## Quick Navigation

### For New Users
1. Start with [Installation Guide](./INSTALLATION.md)
2. Read [Quick Start Guide](./QUICK_START_AFTER_REORGANIZATION.md)
3. Explore [Visualization Overview](./visualizations/README.md)

### For Researchers
- [Heatmap Guide](./visualizations/Heatmap_Guide.md) - Quantitative analysis
- [Network Guide](./visualizations/Network_Guide.md) - Topology exploration
- [Path Finding](./core-features/README.md#path-finding) - Multi-hop connections

### For Developers
- [Technical Documentation](./technical/README.md) - System architecture
- [Core Features](./core-features/README.md) - API and algorithms
- [Archive](./archive/) - Historical changes and fixes

---

## Key Documents

### Getting Started
- **[Installation](./INSTALLATION.md)**: Setup instructions and requirements
- **[Quick Start](./QUICK_START_AFTER_REORGANIZATION.md)**: Get running in 5 minutes
- **[Quickstart Simple Format](./QUICKSTART_SIMPLE_FORMAT.md)**: Using simple edge-list data

### Visualization Guides
- **[Heatmap Guide](./visualizations/Heatmap_Guide.md)**: Complete heatmap documentation
- **[Network Guide](./visualizations/Network_Guide.md)**: Network visualization reference
- **[Sankey Guide](./visualizations/Sankey_Guide.md)**: Flow diagram documentation
- **[3D Skeleton Guide](./visualizations/3D_Skeleton_Guide.md)**: 3D rendering guide

### Core Functionality
- **[Cache System Guide](./core-features/CacheSystem_Guide.md)**: Caching for 10-100x speedup
- **[Path Finding](./core-features/FindAllPath_Documentation.md)**: Multi-hop path discovery
- **[Custom Groups](./core-features/CustomGroups_Feature.md)**: Flexible neuron grouping
- **[Parallel Processing](./core-features/ParallelProcessing_Documentation.md)**: Multi-core acceleration

---

## Documentation by Task

### Finding Connections
| Task | Documentation |
|------|--------------|
| Direct connections | [Main README](../README.md#finddirectpy) |
| Multi-hop paths | [FindAllPath](./core-features/FindAllPath_Documentation.md) |
| Custom neuron groups | [Custom Groups](./core-features/CustomGroups_Feature.md) |
| Forward-only paths | [Forward-Only Guide](./core-features/ForwardOnly_Guide.md) |
| Filter connections | [Connection Filters](./core-features/README.md#filtering) |

### Visualizing Data
| Task | Documentation |
|------|--------------|
| Connection matrix | [Heatmap Guide](./visualizations/Heatmap_Guide.md) |
| Network graph | [Network Guide](./visualizations/Network_Guide.md) |
| Flow diagram | [Sankey Guide](./visualizations/Sankey_Guide.md) |
| 3D anatomy | [3D Skeleton Guide](./visualizations/3D_Skeleton_Guide.md) |

### Optimizing Performance
| Task | Documentation |
|------|--------------|
| Enable caching | [Cache Quick Start](./core-features/CacheSystem_QuickStart.md) |
| Parallel processing | [Parallel Processing Guide](./core-features/ParallelProcessing_Documentation.md) |
| Optimize queries | [Deep Backend Optimizations](./technical/DeepBackendOptimizations.md) |

### Customizing Output
| Task | Documentation |
|------|--------------|
| Custom colors | [Custom Colors Guide](./visualizations/CUSTOM_COLORS_GUIDE.md) |
| Layout algorithms | [Advanced Layout Algorithms](./visualizations/AdvancedLayoutAlgorithms.md) |
| Heatmap clustering | [Heatmap Clustering](./visualizations/HEATMAP_CLUSTERING_FEATURE.md) |
| Edge width control | [Edge Width Scaling](./visualizations/EDGE_WIDTH_SCALING.md) |

---

## Common Workflows

### Workflow 1: Basic Connection Analysis
```
1. Find direct connections (FindDirect.py)
2. Visualize as heatmap
3. Export results
```
**Documentation**: [README](../README.md#basic-functions) → [Heatmap Guide](./visualizations/Heatmap_Guide.md)

### Workflow 2: Pathway Discovery
```
1. Find multi-hop paths (FindAllPath)
2. Visualize as network + Sankey
3. Apply clustering to heatmap
4. Export layouts
```
**Documentation**: [FindAllPath](./core-features/FindAllPath_Documentation.md) → [Network Guide](./visualizations/Network_Guide.md)

### Workflow 3: High-Performance Analysis
```
1. Setup cache (first time)
2. Enable parallel processing
3. Run large-scale queries
4. Generate visualizations
```
**Documentation**: [Cache Setup](./core-features/CacheSystem_QuickStart.md) → [Parallel Processing](./core-features/ParallelProcessing_Documentation.md)

### Workflow 4: Publication Figures
```
1. Generate visualizations
2. Customize colors and layout
3. Export high-resolution PNG/SVG
4. Integrate with other analysis
```
**Documentation**: [Visualization Overview](./visualizations/README.md) → Individual visualization guides

---

## Recent Updates

### Latest Features
- ✨ Multiple clustering algorithms for heatmaps (Ward, Average, Complete, Single)
- ✨ Enhanced edge-list format with flexible column recognition
- ✨ Expanded canvas for better visualization space
- ✨ Reorganized UI controls for better usability

### Recent Improvements
- ⚡ Cache v4 with pair-level optimization
- ⚡ Improved parallel processing with better progress tracking
- 🎨 Enhanced color customization for all visualizations
- 📊 Better heatmap clustering with algorithm selection

---

## Archive

Historical documentation and fixes are in the [archive](./archive/) directory:
- Bug fixes and corrections
- Implementation summaries
- Reorganization history
- Deprecated features

---

## External Resources

### Data Sources
- **NeuPrint**: https://neuprint.janelia.org/
- **FlyEM**: https://www.janelia.org/project-team/flyem
- **Hemibrain Paper**: Scheffer et al. (2020), eLife

### Related Tools
- **NAVIS**: https://navis.readthedocs.io/
- **neuPrint Python**: https://connectome-neuprint.github.io/neuprint-python/
- **FlyWire**: https://flywire.ai/

### Visualization Libraries
- **Plotly**: https://plotly.com/python/
- **Cytoscape.js**: https://js.cytoscape.org/
- **NetworkX**: https://networkx.org/

---

## Getting Help

### Documentation Issues
- Check the specific guide for your task
- Look in [troubleshooting sections](#)
- Review [examples directory](../examples/)

### Technical Support
- [GitHub Issues](https://github.com/Swida-Alba/hemibrain-connectomes-analysis/issues)
- Check [Technical Documentation](./technical/README.md)
- Review [FAQ](#) (coming soon)

### Contributing
- See [Contributing Guide](../CONTRIBUTING.md) (if available)
- Read [Technical Documentation](./technical/README.md)
- Follow coding standards

---

## Documentation Maintenance

### Last Updated
- Major reorganization: 2025-11-05
- Visualization guides created: 2025-11-05
- Test cleanup: 2025-11-05

### Version
Documentation version 3.0 (matches software version)

### Feedback
Documentation feedback welcome via GitHub Issues or pull requests.

---

## Quick Links

**Essential Reading**:
- [Installation](./INSTALLATION.md)
- [Quick Start](./QUICK_START_AFTER_REORGANIZATION.md)
- [Visualization Overview](./visualizations/README.md)

**Visualization Guides**:
- [Heatmap](./visualizations/Heatmap_Guide.md) | [Network](./visualizations/Network_Guide.md) | [Sankey](./visualizations/Sankey_Guide.md) | [3D](./visualizations/3D_Skeleton_Guide.md)

**Core Features**:
- [Cache System](./core-features/CacheSystem_Guide.md) | [Path Finding](./core-features/FindAllPath_Documentation.md) | [Parallel Processing](./core-features/ParallelProcessing_Documentation.md)

**Technical**:
- [Optimization](./technical/DeepBackendOptimizations.md) | [Architecture](./technical/README.md) | [Data Formats](./technical/COLUMN_RECOGNITION_UPDATE.md)

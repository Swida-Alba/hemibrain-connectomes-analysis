# Documentation Structure

**Last Updated:** November 5, 2025

---

## Overview

All documentation files have been organized into the `docs/` folder for better project structure and maintainability.

## 🆕 Latest Features (November 2025)

### Interactive Network Enhancements
- **Edge Filtering:** Hide edges by weight with comparison operators (`<5`, `>100`, etc.)
- **Enhanced Export/Import (v2.0):** Save and restore ALL settings including filters
- **Complete Workflow Support:** Resume work exactly where you left off

See [Quick Reference](#new-features-quick-reference) below.

---

## File Organization

```
drocat/
│
├── README.md                          ← Main entry point with navigation
│
├── docs/                              ← All documentation
│   │
│   ├── CacheSystem_Guide.md          ← 🌟 CONSOLIDATED: Complete cache guide
│   ├── ForwardOnly_Guide.md          ← 🌟 CONSOLIDATED: Complete forward_only guide
│   │
│   ├── CacheSystem_*.md              (9 files - cache system documentation)
│   ├── FindAllPath_*.md              (4 files - pathfinding documentation)
│   ├── ParallelProcessing_*.md       (5 files - parallel processing)
│   ├── FilterBy_*.md                 (2 files - filtering)
│   ├── FolderNaming_*.md             (2 files - folder naming)
│   ├── ConnectionRatio_Filter.md
│   ├── TraversalProbability_EdgeLevelFilter.md
│   ├── PathAnalysis_ZeroWeightFilter.md
│   ├── FindPath_Optimizations_Applied.md
│   ├── ForwardOnly_RealLayer_Implementation.md
│   ├── TargetRealLayer_AccurateImplementation.md
│   ├── Interactive_Network_Dragging_Guide.md
│   ├── NetworkLayout_Documentation.md
│   ├── PathFinding_*.md              (3 files - pathfinding optimization)
│   ├── REQUIREMENTS_CHECKLIST.md
│   ├── DOCUMENTATION_CONSOLIDATION_PLAN.md
│   ├── DOCUMENTATION_REORGANIZATION_SUMMARY.md
│   │
│   └── archive/                      ← Historical bug fixes & corrections
│       ├── BUGFIX_TypeColumn_Merge.md
│       ├── CRITICAL_FIX_Multiprocessing.md
│       ├── CRITICAL_FIX_TargetRealLayer.md
│       ├── CORRECTIONS_SUMMARY.md
│       ├── FindAllPath_CutoffBugFix.md
│       ├── FindAllPath_EdgeHandling_Analysis.md
│       ├── FindAllPath_FolderStructure_Update.md
│       ├── FindAllPath_SankeyFromPaths_Fix.md
│       ├── FindAllPath_Updated_Logic.md
│       └── CacheSystem_InstanceColumn_Fix.md
│
├── [Python files, example scripts, etc.]
└── [Other project files]
```

---

## Documentation Navigation

### 🚀 Start Here

**[README.md](../README.md)** - Main project documentation with:
- Quick start guide
- Basic functions (FindDirect, FindPath)
- Installation instructions
- Complete documentation index

---

### 📖 Core Guides (Consolidated)

#### 1. [CacheSystem_Guide.md](../core-features/CacheSystem_Guide.md) 🌟
**Comprehensive guide to local caching (v3.0)**

Topics covered:
- Quick start and setup
- How caching works (fetch-once, filter-many strategy)
- Database architecture (registry, index, connections)
- Storage optimization (40-60% savings)
- Cache management and troubleshooting
- Best practices

**Consolidates:** 4 cache-related documents into one comprehensive guide

#### 2. [ForwardOnly_Guide.md](../core-features/ForwardOnly_Guide.md) 🌟
**Complete guide to forward_only parameter**

Topics covered:
- Three functions: querying strategy, path validation, visualization filtering
- Real layer vs appearance layer concepts
- Path validation rules
- Visualization filtering logic
- Usage examples and troubleshooting

**Consolidates:** 4 forward_only documents into one comprehensive guide

---

### 🔧 Feature Documentation

#### Cache System (9 files)
- **[CacheSystem_Guide.md](../core-features/CacheSystem_Guide.md)** - Complete guide
- [CacheSystem_v3_DatabaseArchitecture.md](../core-features/CacheSystem_v3_DatabaseArchitecture.md) - Technical architecture
- [CacheSystem_QuickStart.md](../core-features/CacheSystem_QuickStart.md) - Quick reference
- [CacheSystem_CompleteDataset.md](../core-features/CacheSystem_CompleteDataset.md) - Handling type=None neurons
- [CacheSystem_StorageOptimization.md](../core-features/CacheSystem_StorageOptimization.md) - Storage details
- [CacheSystem_v4_Complete.md](../core-features/CacheSystem_v4_Complete.md) - Future version
- [CacheSystem_v4_Implementation.md](../core-features/CacheSystem_v4_Implementation.md) - v4 implementation
- [CacheSystem_v4_PairLevel_Proposal.md](../core-features/CacheSystem_v4_PairLevel_Proposal.md) - v4 proposal

#### PathFinding (7 files)
- [FindAllPath_Documentation.md](../core-features/FindAllPath_Documentation.md) - Main pathfinding docs
- [FindAllPath_CachingImprovement.md](../core-features/FindAllPath_CachingImprovement.md) - Cache integration
- [FindAllPath_MultipleConnections_Example.md](../core-features/FindAllPath_MultipleConnections_Example.md) - Examples
- [FindPath_Optimizations_Applied.md](../core-features/FindPath_Optimizations_Applied.md) - Applied optimizations
- [PathFinding_DynamicETA.md](../core-features/PathFinding_Methods.md) - ETA calculation
- [PathFinding_RealisticEstimation.md](../core-features/PathFinding_Methods.md) - Time estimation
- [PathFinding_Optimization.md](../core-features/PathFinding_Methods.md) - General optimization

#### Forward-Only Mode (4 files)
- **[ForwardOnly_Guide.md](../core-features/ForwardOnly_Guide.md)** - Complete guide
- [ForwardOnly_RealLayer_Implementation.md](../core-features/ForwardOnly_RealLayer_Implementation.md) - Real layer logic
- [TargetRealLayer_AccurateImplementation.md](TargetRealLayer_AccurateImplementation.md) - Target handling
- [FindAllPath_ForwardOnly_Explanation.md](../core-features/FindAllPath_ForwardOnly_Explanation.md) - Explanation
- [FindAllPath_TrueVsFalse_Final.md](../core-features/FindAllPath_TrueVsFalse_Final.md) - Comparison

#### Parallel Processing (5 files)
- ParallelProcessing_Documentation.md - Main documentation
- ParallelProcessing_QuickReference.md - Quick reference
- ParallelProcessing_Implementation_Summary.md - Implementation
- ParallelProcessing_ProgressTracking.md - Progress tracking
- ParallelProcessing_ImprovedProgress_v2.md - Latest progress

#### Filtering & Configuration (5 files)
- [ConnectionRatio_Filter.md](../core-features/ConnectionRatio_Filter.md) - Connection ratio filtering
- [TraversalProbability_EdgeLevelFilter.md](../core-features/TraversalProbability_EdgeLevelFilter.md) - Traversal probability
- [FilterBy_Feature.md](../core-features/FilterBy_Feature.md) - Filter features
- [FilterBy_QuickRef.md](../core-features/FilterBy_QuickRef.md) - Quick reference
- [PathAnalysis_ZeroWeightFilter.md](../core-features/PathAnalysis_ZeroWeightFilter.md) - Zero weight handling

#### Visualization & Interaction (6 files) 🆕
- **[NETWORK_QUICKREF.md](../visualizations/NETWORK_QUICKREF.md)** - Quick reference for all network features 🌟
- **[NETWORK_EDGE_FILTER.md](../visualizations/NETWORK_EDGE_FILTER.md)** - Edge filtering guide (v2.0) 🆕
- **[NETWORK_EXPORT_IMPORT.md](../visualizations/NETWORK_EXPORT_IMPORT.md)** - Export/import with settings (v2.0) 🆕
- [NetworkLayout_Documentation.md](../visualizations/NetworkLayout_Documentation.md) - Network layouts
- [Interactive_Network_Dragging_Guide.md](../visualizations/Interactive_Network_Dragging_Guide.md) - Interactive features
- [NETWORK_INTERACTIVE_EDITING.md](../visualizations/Interactive_Network_Dragging_Guide.md) - Edit mode guide

#### Folder Naming (2 files)
- [FolderNaming_Convention.md](FolderNaming_Convention.md) - Naming convention
- [FolderNaming_DecimalNotation_Fix.md](FolderNaming_DecimalNotation_Fix.md) - Decimal notation

---

### 📋 Project Documentation (3 files)
- [REQUIREMENTS_CHECKLIST.md](REQUIREMENTS_CHECKLIST.md) - Requirements tracking
- [DOCUMENTATION_CONSOLIDATION_PLAN.md](DOCUMENTATION_CONSOLIDATION_PLAN.md) - Consolidation plan
- [DOCUMENTATION_REORGANIZATION_SUMMARY.md](DOCUMENTATION_REORGANIZATION_SUMMARY.md) - Recent changes

---

### 🗄️ Historical Archive (10 files)

Located in `docs/archive/` - preserved for reference:

**Bug Fixes:**
- [BUGFIX_TypeColumn_Merge.md](BUGFIX_TypeColumn_Merge.md)
- [CRITICAL_FIX_Multiprocessing.md](CRITICAL_FIX_Multiprocessing.md)
- [CRITICAL_FIX_TargetRealLayer.md](CRITICAL_FIX_TargetRealLayer.md)
- [CacheSystem_InstanceColumn_Fix.md](CacheSystem_InstanceColumn_Fix.md)

**Implementation Details:**
- [CORRECTIONS_SUMMARY.md](CORRECTIONS_SUMMARY.md)
- [FindAllPath_CutoffBugFix.md](FindAllPath_CutoffBugFix.md)
- [FindAllPath_EdgeHandling_Analysis.md](FindAllPath_EdgeHandling_Analysis.md)
- [FindAllPath_FolderStructure_Update.md](FindAllPath_FolderStructure_Update.md)
- [FindAllPath_SankeyFromPaths_Fix.md](FindAllPath_SankeyFromPaths_Fix.md)
- [FindAllPath_Updated_Logic.md](FindAllPath_Updated_Logic.md)

---

## Quick Links by Topic

### Getting Started
1. [README.md](../README.md) - Start here
2. [CacheSystem_Guide.md](../core-features/CacheSystem_Guide.md) - Enable caching for speed
3. [FindAllPath_Documentation.md](../core-features/FindAllPath_Documentation.md) - Find paths

### Performance
- [CacheSystem_Guide.md](../core-features/CacheSystem_Guide.md) - 10-100x speedup
- ParallelProcessing_Documentation.md - Multi-core acceleration

### Path Analysis
- [ForwardOnly_Guide.md](../core-features/ForwardOnly_Guide.md) - Path validation
- [FindAllPath_Documentation.md](../core-features/FindAllPath_Documentation.md) - Pathfinding

### Filtering
- [ConnectionRatio_Filter.md](../core-features/ConnectionRatio_Filter.md) - Connection ratio
- [TraversalProbability_EdgeLevelFilter.md](../core-features/TraversalProbability_EdgeLevelFilter.md) - Probability
- [FilterBy_Feature.md](../core-features/FilterBy_Feature.md) - Filter features

### Visualization & Interaction 🆕
- **[NETWORK_QUICKREF.md](../visualizations/NETWORK_QUICKREF.md)** - Quick reference for all features
- **[NETWORK_EDGE_FILTER.md](../visualizations/NETWORK_EDGE_FILTER.md)** - Hide edges by weight
- **[NETWORK_EXPORT_IMPORT.md](../visualizations/NETWORK_EXPORT_IMPORT.md)** - Save/restore with settings
- [NetworkLayout_Documentation.md](../visualizations/NetworkLayout_Documentation.md) - Layout algorithms
- [Interactive_Network_Dragging_Guide.md](../visualizations/Interactive_Network_Dragging_Guide.md) - Interactive controls
- [NETWORK_INTERACTIVE_EDITING.md](../visualizations/Interactive_Network_Dragging_Guide.md) - Edit mode

---

## New Features Quick Reference

### Edge Filtering (v2.0)

**What:** Hide edges based on weight values using simple expressions

**Syntax:**
```
<5              → Hide edges with weight < 5
>100            → Hide edges with weight > 100
1, 5, 10        → Hide edges with specific weights
<10, >90        → Combine multiple conditions
```

**All operators:** `<`, `>`, `<=`, `>=`, `==`, `!=`

**Documentation:** [NETWORK_EDGE_FILTER.md](../visualizations/NETWORK_EDGE_FILTER.md)

### Enhanced Export/Import (v2.0)

**What:** Save and restore complete visualization state

**New in v2.0:**
- ✅ Edge filter expressions
- ✅ Scaling method (Linear/Log/Sqrt)
- ✅ All slider values (font, node size, arrow size)
- ✅ Auto-applies settings on import

**Workflow:**
```
1. Set filter: <5
2. Adjust settings
3. Export → network_graph_2025-11-05.json
4. [Later] Import → Everything restored!
```

**Documentation:** [NETWORK_EXPORT_IMPORT.md](../visualizations/NETWORK_EXPORT_IMPORT.md)

### Complete Quick Reference

**All features in one place:** [NETWORK_QUICKREF.md](../visualizations/NETWORK_QUICKREF.md)

---

## Recent Changes

**November 5, 2025:**
1. ✅ Added edge filtering by weight with comparison operators
2. ✅ Enhanced export/import to v2.0 with settings preservation
3. ✅ Created comprehensive feature documentation:
   - NETWORK_EDGE_FILTER.md (complete filtering guide)
   - NETWORK_EXPORT_IMPORT.md (export/import workflows)
   - NETWORK_QUICKREF.md (quick reference card)
4. ✅ Updated DOCUMENTATION_STRUCTURE.md with new features

**October 27, 2025:**
1. ✅ Moved all documentation to `docs/` folder (37 files)
2. ✅ Created consolidated guides (CacheSystem_Guide.md, ForwardOnly_Guide.md)
3. ✅ Archived historical bug fixes to `docs/archive/` (10 files)
4. ✅ Updated README.md with comprehensive navigation section
5. ✅ Fixed all documentation links to point to `docs/` folder

**Result:**
- Root directory: 1 markdown file (README.md only)
- Documentation folder: 37 organized files + consolidated guides
- Archive folder: 10 historical documents
- Clean, organized, easy to navigate structure

---

## Statistics

**Total Documentation Files:** 41
- Main README: 1
- Core guides (consolidated): 2
- Feature documentation: 30 (🆕 +3 network features)
- Project documentation: 3
- Historical archive: 10

**Organization:**
- ✅ Single entry point (README.md)
- ✅ Clear topic-based organization
- ✅ Consolidated guides for major topics
- ✅ Historical context preserved
- ✅ All links updated and functional

---

## Maintenance

### Adding New Documentation

1. Create file in appropriate subfolder of `docs/`
2. Add link to README.md navigation section
3. Update this DOCUMENTATION_STRUCTURE.md

### Updating Existing Documentation

1. Edit the file in `docs/`
2. Update README.md if needed
3. Check for broken cross-references

### Archiving Old Documentation

1. Move to `docs/archive/`
2. Update README.md navigation
3. Document reason in DOCUMENTATION_REORGANIZATION_SUMMARY.md

---

**For questions or suggestions about documentation organization, see:**
- [DOCUMENTATION_CONSOLIDATION_PLAN.md](DOCUMENTATION_CONSOLIDATION_PLAN.md)
- [DOCUMENTATION_REORGANIZATION_SUMMARY.md](DOCUMENTATION_REORGANIZATION_SUMMARY.md)

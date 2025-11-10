# Layout Algorithm Quick Reference

## 🎯 Which Layout Should I Use?

```
┌─────────────────────────────────────────────────────────┐
│                 QUICK DECISION TREE                      │
└─────────────────────────────────────────────────────────┘

Is your network hierarchical (pathways with layers)?
│
├─ YES → Use DAGRE ⭐⭐⭐⭐⭐ (Default)
│        → Alternative: KLay ⭐⭐⭐⭐
│        → Fast option: Breadth-First ⭐⭐⭐
│
└─ NO → Is it non-hierarchical (recurrent/feedback)?
        │
        ├─ YES → Use fCoSE ⭐⭐⭐⭐⭐
        │        → Alternative: CoSE-Bilkent ⭐⭐⭐⭐
        │        → Fast option: CoSE ⭐⭐⭐
        │
        └─ UNSURE → Try Dagre first, then fCoSE
                    → Compare visually
```

---

## 📊 Layout Comparison Chart

| Layout | Stars | Speed | Crossings | Best For |
|--------|-------|-------|-----------|----------|
| **Dagre** | ⭐⭐⭐⭐⭐ | Fast | Minimal | Pathways |
| **fCoSE** | ⭐⭐⭐⭐⭐ | Fast | Low | General |
| **KLay** | ⭐⭐⭐⭐ | Medium | Very Low | Complex Hierarchical |
| **CoSE-Bilkent** | ⭐⭐⭐⭐ | Medium | Low | High Quality |
| **Breadth-First** | ⭐⭐⭐ | V.Fast | Medium | Simple Trees |
| **CoSE** | ⭐⭐⭐ | Fast | Medium | Quick Force |
| **Circular** | ⭐⭐ | V.Fast | Many | Small Networks |
| **Grid** | ⭐⭐ | V.Fast | Medium | Regular Structure |
| **Concentric** | ⭐⭐ | V.Fast | Medium | Hub Networks |

---

## ⌨️ Keyboard Shortcuts

```
H    - Hide selected nodes (Shift+Click to select multiple)
E    - Hide selected edges
L    - Toggle label position (center ↔ outside)
```

---

## 🎨 Visual Guide

### Dagre (Recommended for Pathways)
```
     Source
       ↓
   Intermediate
       ↓
     Target
```
- Layers aligned vertically
- Minimal edge crossings
- Clean hierarchy

### fCoSE (Best for Non-Hierarchical)
```
    A ←→ B
    ↓ ⤢  ↓
    C ←→ D
```
- Natural clustering
- Good for feedback loops
- Balanced layout

### Circular (Small Networks Only)
```
      A
    ↙   ↘
   D     B
    ↖   ↗
      C
```
- All nodes visible
- Many crossings
- Use for <20 nodes

---

## 🚀 Quick Start

### In FindPath.py:
```python
fc = FindNeuronConnection(
    network_layout='hierarchical',  # Uses Dagre now!
    ...
)
```

### In Generated HTML:
1. Look for **🔧 Layout Algorithm** dropdown
2. Select layout from menu
3. Graph re-arranges automatically

---

## 💡 Pro Tips

### 1. Start with Dagre
Always try **Dagre** first for neural pathways - it's specifically designed for hierarchical graphs and minimizes crossings.

### 2. Compare Layouts
Click through 2-3 layouts to see which is clearest for your specific network.

### 3. Save Your Layout
After manually adjusting node positions:
- Click **💾 Save** button
- Positions saved to browser
- Click **📂 Load** to restore

### 4. Use Keyboard Shortcuts
- Hide clutter with `H` (nodes) and `E` (edges)
- Toggle labels with `L`
- Much faster than clicking

### 5. Export High Quality
- Set export scale to 3-4x
- Use SVG for publications
- Use PNG for presentations

---

## 🔧 Troubleshooting

### Too many crossings?
→ Try: Dagre → KLay → fCoSE (in that order)

### Layout too slow?
→ Try: Breadth-First → CoSE → Dagre

### Nodes overlapping?
→ Increase node spacing (adjust sliders)
→ Try Grid or Circular for small networks

### Can't see all connections?
→ Try Circular layout
→ Adjust zoom level
→ Hide less important nodes

---

## 📝 Common Use Cases

### Neural Pathways (FindPath.py)
**Best:** Dagre ⭐⭐⭐⭐⭐
- Shows clear source → intermediate → target flow
- Minimizes edge crossings between layers
- Easy to trace paths

### Direct Connections (FindDirect.py)
**If source ≠ target:** Breadth-First ⭐⭐⭐
**If source = target:** fCoSE ⭐⭐⭐⭐⭐

### Small Networks (<20 nodes)
**Best:** Circular ⭐⭐ or Dagre ⭐⭐⭐⭐⭐
- Both work well
- Circular shows all connections
- Dagre still organizes hierarchy

### Large Networks (>200 nodes)
**Best:** Dagre ⭐⭐⭐⭐⭐ or fCoSE ⭐⭐⭐⭐⭐
- Both scale well
- CoSE-Bilkent may be slow
- Avoid Circular and Grid

### Publication Figures
**Best:** Dagre ⭐⭐⭐⭐⭐ or CoSE-Bilkent ⭐⭐⭐⭐
- Highest quality
- Clean appearance
- Export at 3-4x scale

---

## 🎓 Algorithm Cheat Sheet

| Want... | Use... |
|---------|--------|
| **Minimal crossings** | Dagre |
| **Fastest layout** | Breadth-First |
| **Highest quality** | CoSE-Bilkent |
| **Best for pathways** | Dagre |
| **Best for general** | fCoSE |
| **Good for papers** | Dagre or CoSE-Bilkent |
| **Quick preview** | Breadth-First or CoSE |

---

## 📖 More Info

See **AdvancedLayoutAlgorithms.md** for:
- Detailed algorithm descriptions
- Performance benchmarks
- Technical configurations
- Advanced usage examples

---

**Last Updated**: October 31, 2025
**Version**: 2.1.0

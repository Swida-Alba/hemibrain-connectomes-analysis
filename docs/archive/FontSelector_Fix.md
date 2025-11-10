# Font Selector Fix - October 27, 2025

## Problem
The font selector dropdown wasn't working properly for all fonts:
- Some fonts weren't being applied correctly
- Font names with spaces needed proper quoting
- Edge labels weren't getting font updates
- Fallback fonts weren't configured

## Root Causes

### 1. **Incorrect Font Value Format**
The original dropdown used full CSS font stacks like:
```html
<option value="Arial, sans-serif">Arial</option>
<option value="'Helvetica Neue', Helvetica, sans-serif">Helvetica</option>
```

This caused issues because:
- Mixed quoting styles (some with quotes, some without)
- Cytoscape.js needs consistent font-family format
- No clear mapping between display name and actual font

### 2. **Missing Font Fallbacks**
Fonts need proper fallback stacks for cross-platform compatibility:
- Windows might not have fonts that macOS has (and vice versa)
- Web fonts need generic fallbacks (sans-serif, serif, monospace)

### 3. **Edge Labels Not Updated**
The `updateFont()` function only updated node labels:
```javascript
cy.style()
    .selector('node')
    .style('font-family', fontFamily)
    .update();
```

Edge hover tooltips use their own font-family style and weren't being updated.

### 4. **No Default Font in Styles**
The initial Cytoscape styles didn't include `font-family`, relying on browser defaults.

## Solution

### 1. **Simplified Font Values**
Changed dropdown to use clean font names:
```html
<option value="Arial">Arial</option>
<option value="'Segoe UI'">Segoe UI</option>
<option value="'Times New Roman'">Times New Roman</option>
<option value="'Courier New'">Courier New</option>
```

Single fonts that need quotes (multi-word names) are quoted; single-word fonts are not.

### 2. **Smart Fallback System**
The `updateFont()` function now builds proper fallback stacks:

```javascript
function updateFont() {
    const fontFamily = document.getElementById('fontFamily').value;
    
    // Create fallback font stack based on selection
    let fontStack = fontFamily;
    
    // Add appropriate fallbacks for specific fonts
    if (fontFamily.includes('Segoe UI')) {
        fontStack = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
    } else if (fontFamily.includes('Times')) {
        fontStack = "'Times New Roman', Times, Georgia, serif";
    } else if (fontFamily.includes('Courier')) {
        fontStack = "'Courier New', Courier, 'Lucida Console', monospace";
    }
    // ... more fallbacks ...
    
    // Update both node and edge font families
    cy.style()
        .selector('node')
        .style('font-family', fontStack)
        .selector('edge')
        .style('font-family', fontStack)
        .update();
}
```

### 3. **Font Fallback Chains**

| Selected Font      | Fallback Stack                                           |
|--------------------|----------------------------------------------------------|
| Segoe UI           | 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif         |
| Arial              | Arial, Helvetica, sans-serif                             |
| Helvetica          | Helvetica, Arial, sans-serif                             |
| Times New Roman    | 'Times New Roman', Times, Georgia, serif                 |
| Courier New        | 'Courier New', Courier, 'Lucida Console', monospace      |
| Georgia            | Georgia, 'Times New Roman', serif                        |
| Verdana            | Verdana, Geneva, sans-serif                              |
| Trebuchet MS       | 'Trebuchet MS', 'Lucida Grande', Tahoma, sans-serif      |
| Comic Sans MS      | 'Comic Sans MS', 'Comic Sans', cursive, sans-serif       |
| Impact             | Impact, 'Arial Black', sans-serif                        |
| Monospace          | 'Courier New', Courier, monospace                        |
| Sans-serif         | Arial, Helvetica, sans-serif                             |
| Serif              | Georgia, 'Times New Roman', serif                        |

### 4. **Apply to Both Nodes and Edges**
The function now updates both selectors in one operation:
```javascript
cy.style()
    .selector('node')
    .style('font-family', fontStack)
    .selector('edge')  // Also update edge labels
    .style('font-family', fontStack)
    .update();
```

### 5. **Default Font in Styles**
Added default font-family to initial Cytoscape styles:

**Node styles:**
```javascript
{
    selector: 'node',
    style: {
        // ... other styles ...
        'font-family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    }
}
```

**Edge styles:**
```javascript
{
    selector: 'edge',
    style: {
        // ... other styles ...
        'font-family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    }
}
```

## Updated Font Options

The dropdown now includes 13 fonts:

### Professional Fonts
1. **Segoe UI** (default) - Modern, clean, Windows/Office standard
2. **Arial** - Universal, highly readable
3. **Helvetica** - Classic, professional
4. **Verdana** - Optimized for screen readability
5. **Georgia** - Elegant serif font
6. **Trebuchet MS** - Humanist sans-serif

### Special Purpose Fonts
7. **Times New Roman** - Traditional serif, formal documents
8. **Courier New** - Monospace, code-like appearance
9. **Comic Sans MS** - Casual, informal (yes, it's there!)
10. **Impact** - Bold, attention-grabbing

### Generic Families
11. **Monospace** - Fixed-width coding font
12. **Sans-serif** - Clean, modern generic
13. **Serif** - Traditional generic

## Testing

To test the font selector:

1. **Open network visualization**
   ```bash
   python PlotPath.py
   ```

2. **Open Style Panel** (⚙️ icon in top-right)

3. **Test each font:**
   - Select font from dropdown
   - Font applies immediately (no need to click "Apply Changes")
   - Check both node labels and edge hover tooltips
   - Verify fallback works on different systems

4. **Console debugging:**
   - Open browser DevTools (F12)
   - Select fonts and check console output:
     ```
     Font updated: 'Segoe UI' → 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif
     ```

## Cross-Platform Compatibility

The fallback system ensures fonts work across:
- **Windows**: Segoe UI, Arial, Times New Roman, Courier New, Verdana, Trebuchet MS, Georgia, Impact, Comic Sans MS
- **macOS**: Helvetica, Geneva, Lucida Grande, Times, Courier
- **Linux**: Liberation Sans, DejaVu Sans, Liberation Mono
- **All platforms**: Generic families (sans-serif, serif, monospace) always work

## Files Modified
- `vispath.py`
  - Font dropdown HTML (lines ~773-786)
  - Node style default font (lines ~857)
  - Edge style default font (lines ~906)
  - `updateFont()` function (lines ~1185-1227)

## Benefits

1. ✅ **All fonts now work** - Proper quoting and fallbacks
2. ✅ **Instant updates** - No need to click "Apply Changes"
3. ✅ **Cross-platform** - Works on Windows, macOS, Linux
4. ✅ **Edge labels included** - Hover tooltips use same font
5. ✅ **Graceful degradation** - Falls back to similar fonts if unavailable
6. ✅ **Console feedback** - Debug output shows font stack being applied

## Related Issues
- Original issue: "not all the fonts work"
- Root cause: Missing fallbacks + improper quoting + edge labels not updated
- Fixed: Complete font system overhaul with smart fallbacks

---

**Author**: Font system fix implemented 2025-10-27  
**Related docs**: `VisualizationFixes_2025-10-27.md`

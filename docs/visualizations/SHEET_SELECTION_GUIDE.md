# Sheet Selection Guide

## Overview

The VisualizePath class now includes **GUI-based sheet selection dialogs** for Excel files, with intelligent auto-detection and flexible manual selection.

## Features

### 1. **File Picker Dialog** 🗂️
- Opens native file browser when `path_file=None`
- Supports `.csv`, `.xlsx`, and `.xls` files
- Cross-platform compatible (macOS, Windows, Linux)
- Platform-specific window focus handling

### 2. **Smart Sheet Selection** ✨

#### Auto-Detection
Automatically detects common sheet names:
- `path_type`
- `path_bodyId`
- `path_block`
- `paths`

#### Combined Selection Dialog
When file picker is used:
- **Pre-selects** auto-detected sheet with ✓ marker
- Shows all available sheets with metadata (rows, columns)
- User can accept default or select different sheet
- **One dialog** instead of confirm + select

#### Silent Auto-Selection
When direct file path provided:
- Silently selects common sheet names
- No dialog shown for scripted workflows
- Maintains backward compatibility

### 3. **Windows Compatibility** 🪟
- Platform-specific `tkinter` settings
- `root.lift()` and `focus_force()` for Windows
- `os.path.normpath()` for cross-platform paths
- Try/except handling for platform differences

### 4. **Terminal Fallback** 💻
When `tkinter` is unavailable:
- Falls back to terminal-based selection
- Same functionality, different interface
- Press Enter to accept auto-detected sheet
- Type number or name to select different sheet

## Usage Examples

### Example 1: File Picker (Interactive)
```python
from vispath import VisualizePath

# Opens file picker, then shows sheet selection dialog
vis = VisualizePath(
    path_file=None,  # Triggers file picker
    output_folder='./output'
)
```

**User Experience:**
1. File picker opens → select Excel file
2. Sheet selection dialog appears
3. Auto-detected sheet is pre-selected with ✓
4. Press OK to accept, or select different sheet
5. Done!

### Example 2: Direct Path (Automatic)
```python
from vispath import VisualizePath

# Auto-selects sheet silently (no dialogs)
vis = VisualizePath(
    path_file='./data/pathways.xlsx',  # Direct path
    output_folder='./output'
)
```

**User Experience:**
1. File loads automatically
2. Sheet auto-detected and selected
3. No dialogs shown
4. Done!

### Example 3: Manual Sheet Override
```python
from vispath import VisualizePath

# Explicitly specify sheet name
vis = VisualizePath(
    path_file='./data/pathways.xlsx',
    sheet_name='custom_sheet',  # Override auto-detection
    output_folder='./output'
)
```

## Dialog Screenshots (Conceptual)

### Sheet Selection Dialog
```
┌─────────────────────────────────────────────────────────┐
│ Select Excel Sheet                                   × │
├─────────────────────────────────────────────────────────┤
│ Auto-detected: 'path_type' (pre-selected)              │
│ Press OK to use it, or select a different sheet:       │
│                                                         │
│ ┌─────────────────────────────────────────────────┐ ▲ │
│ │ ✓ path_type (150 rows, 8 cols) [Auto-detected] │ █ │
│ │   random_data (50 rows, 3 cols)                 │ █ │
│ │   metadata (10 rows, 2 cols)                    │ █ │
│ │   backup_paths (150 rows, 8 cols)               │ ▼ │
│ └─────────────────────────────────────────────────┘   │
│                                                         │
│ Tip: Double-click or press OK to select                │
│                                                         │
│                           [ OK ] [ Cancel ]             │
└─────────────────────────────────────────────────────────┘
```

## Behavior Summary

| Scenario | File Picker | Sheet Selection | User Action |
|----------|-------------|-----------------|-------------|
| `path_file=None` | ✅ Opens | ✅ Shows dialog with default | Accept or change |
| Direct path + auto-detected | ❌ No | ❌ No | None (automatic) |
| Direct path + no auto-detect | ❌ No | ✅ Shows all sheets | Select one |
| `sheet_name` specified | ❌ No | ❌ No | None (uses specified) |

## Benefits

### User-Friendly
- ✅ **GUI dialogs** instead of terminal input
- ✅ **Visual feedback** with sheet metadata
- ✅ **Pre-selected defaults** for quick acceptance
- ✅ **Flexible override** without multiple steps

### Developer-Friendly
- ✅ **Backward compatible** with existing code
- ✅ **No breaking changes** to API
- ✅ **Silent operation** for scripts
- ✅ **Terminal fallback** when no GUI available

### Cross-Platform
- ✅ **macOS** - Native file dialogs
- ✅ **Windows** - Proper window focus
- ✅ **Linux** - tkinter support
- ✅ **Headless** - Terminal fallback

## Testing

Run the test script to try both scenarios:

```bash
python test_sheet_confirmation.py
```

Choose option 1 to test file picker + sheet selection, or option 3 to test both interactive and automatic modes.

## Implementation Details

### Key Methods

1. **`_select_file()`**
   - Opens file picker dialog
   - Cross-platform window handling
   - Returns selected file path

2. **`_select_sheet(excel_file, auto_confirm=False)`**
   - Main sheet selection logic
   - `auto_confirm=True` → Shows selection dialog
   - `auto_confirm=False` → Silent auto-selection

3. **`_prompt_sheet_selection(sheet_names, excel_file, default_sheet=None)`**
   - GUI dialog for sheet selection
   - Pre-selects `default_sheet` if provided
   - Shows metadata for each sheet
   - Terminal fallback available

4. **`_confirm_sheet_selection(suggested_sheet, all_sheets, excel_file)`**
   - Wrapper that calls `_prompt_sheet_selection` with default
   - Used when file picker triggers auto-confirm

### Auto-Confirm Flow

```python
# In _load_data():
file_picker_used = False

if path_file is None or not exists:
    path_file = _select_file()  # Open file picker
    file_picker_used = True

if file_picker_used:
    sheet_name = _select_sheet(excel_file, auto_confirm=True)  # Show dialog
elif sheet_name is None:
    sheet_name = _select_sheet(excel_file, auto_confirm=False)  # Auto-select
```

## Future Enhancements

Potential improvements:
- 📊 Preview sheet data in dialog
- 🔍 Search/filter sheets by name
- 📝 Remember last selected sheet
- 🎨 Custom dialog themes
- 📱 Mobile/web interface option

---

**Version:** 2.0  
**Last Updated:** October 28, 2025  
**Platform Support:** macOS, Windows, Linux

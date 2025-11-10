# Layout Persistence Feature

## Overview

The network visualizations now include a **Layout Persistence** feature that allows you to save and restore all your custom adjustments to the visualization.

## Features

### 💾 Save Layout
- Saves the current state to **browser localStorage**
- Persists across browser sessions (stays saved even after closing browser)
- Automatically uses unique storage key based on filename

### 📂 Load Layout
- Restores previously saved layout from localStorage
- Shows timestamp of when layout was saved
- Restores all positions, colors, and settings

### 📤 Export JSON
- Downloads layout configuration as a `.json` file
- Can be shared with collaborators
- Can be backed up or version-controlled

### 📥 Import JSON
- Upload a previously exported layout file
- Applies all settings from the file
- Automatically saves to localStorage after import

## What Gets Saved

The following state is preserved:

### Node Properties
- **Positions** (x, y coordinates)
- **Colors** (custom color assignments)
- **Visibility** (hidden/shown state)

### Edge Properties
- **Visibility** (hidden/shown state)

### View State
- **Zoom level**
- **Pan position** (viewport center)
- **Label visibility** (on/off)

### Control Settings
- **Edge width** slider value
- **Edge width scale** method (linear/log/sqrt/none)
- **Arrow size** slider value
- **Font size** slider value
- **Node size** slider value

### Metadata
- **Timestamp** (when saved)
- **Graph name** (for unique identification)

## Usage

### Basic Workflow

1. **Adjust your visualization**
   - Move nodes around to desired positions
   - Change colors using the color palette
   - Hide/show nodes and edges as needed
   - Adjust font sizes, edge widths, etc.

2. **Save your work**
   - Click **💾 Save** button
   - See confirmation: "✓ Layout saved to browser"
   - Your layout is now persisted in browser storage

3. **Reload anytime**
   - Open the same HTML file
   - Click **📂 Load** button
   - See confirmation with save timestamp
   - All your adjustments are restored

### Sharing Layouts

1. **Export to file**
   - Click **📤 Export** button
   - Downloads `network_layout_<filename>.json`
   - Send this file to collaborators

2. **Import from file**
   - Collaborator clicks **📥 Import** button
   - Selects your `.json` file
   - Layout is automatically applied and saved

## Technical Details

### Storage

- Uses browser's **localStorage** API
- Storage key: `cytoscape_network_layout`
- Data format: JSON
- Size limit: ~5-10 MB (browser dependent, typically sufficient for hundreds of nodes)

### Browser Compatibility

- ✅ Chrome, Edge, Firefox, Safari (modern versions)
- ✅ Works offline (no internet needed after initial page load)
- ❌ Incognito/Private mode (localStorage cleared on exit)

### File Format

JSON structure:
```json
{
  "positions": [{"id": "neuron1", "position": {"x": 100, "y": 200}}, ...],
  "colors": [{"id": "neuron1", "color": "#ff0000"}, ...],
  "visibility": [{"id": "neuron1", "visible": true, "hidden": false}, ...],
  "edgeVisibility": [{"id": "edge1", "visible": true, "hidden": false}, ...],
  "zoom": 1.5,
  "pan": {"x": 0, "y": 0},
  "labelsVisible": true,
  "edgeWidth": "3",
  "edgeWidthScale": "log_2",
  "arrowSize": "9",
  "fontSize": "12",
  "nodeSize": "40",
  "timestamp": "2025-10-29T10:30:00.000Z",
  "graphName": "network_selected_paths"
}
```

### Status Messages

| Message | Meaning |
|---------|---------|
| ✓ Layout saved to browser | Save successful |
| ✓ Layout loaded from \<date> | Load successful with timestamp |
| ✓ Layout exported as JSON | Export successful |
| ✓ Layout imported from \<filename> | Import successful |
| No saved layout found | No previous save in localStorage |
| No layout to export. Save first! | Must save before exporting |
| ✗ Invalid layout file | JSON file format incorrect |
| ✗ Save/Load/Export/Import failed: \<error> | Error occurred |

## Best Practices

### 1. Save Frequently
- Save after major layout changes
- Save before experimenting with new adjustments
- Export important layouts as backup

### 2. Name Your Exports
- Default name: `network_layout_<original_filename>.json`
- Rename exported files with descriptive names
- Example: `network_layout_L3_to_MeVPMe_final_2025.json`

### 3. Version Control
- Export layouts before making major changes
- Keep multiple versions with date suffixes
- Store in project folder for team collaboration

### 4. Browser Considerations
- Each browser has separate localStorage
- Different browsers won't share saved layouts
- Use Export/Import to transfer between browsers

### 5. Backup Important Work
- Export to JSON file for permanent backup
- localStorage can be cleared by browser settings
- JSON files are portable and version-controllable

## Troubleshooting

### Layout not loading?
- Check status message for errors
- Verify you're opening the same HTML file
- Try exporting and re-importing

### Positions slightly off?
- May occur if window size changed significantly
- Click "Fit to Screen" to recenter
- Re-save after adjusting

### Can't save in Incognito mode?
- localStorage disabled in private browsing
- Use Export instead to save to file
- Open in regular browser window

### Shared layout looks different?
- Different screen sizes may affect initial view
- Collaborator should click "Fit to Screen"
- Relative positions will be preserved

## Future Enhancements

Possible future additions:
- Multiple save slots (save different versions)
- Auto-save every N minutes
- Undo/redo functionality
- Layout comparison view
- Cloud storage integration

## Implementation Status

- ✅ **Network Visualization** - Fully implemented
- ⏳ **Sankey Diagram** - Planned (will be added after testing network version)

## Feedback

If you encounter issues or have suggestions for this feature, please open an issue on the GitHub repository.

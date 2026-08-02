# Settings - Instruction

## Dataset Availability

Click **Refresh** to query the NeuPrint server for available datasets.
Requires a valid NeuPrint token.

## API Tokens

- **NeuPrint Token**: required for all NeuPrint datasets. Get it at
  neuprint.janelia.org/account and click Save.
- **CAVE Token**: required for FlyWire FAFB/BANC datasets (which also need
  manually downloaded data files, see the guides in the panel).

## Output Settings

The default output directory is pre-filled in every tool panel; change it
here and click **Save Default Directory**.

## Dataset Preparation

NeuPrint datasets download automatically on first use. FlyWire FAFB/BANC
require downloading the neuron-table CSV from codex.flywire.ai and placing
it in `datasets/flywire_FAFB_v783/` (etc.), plus a CAVE token.

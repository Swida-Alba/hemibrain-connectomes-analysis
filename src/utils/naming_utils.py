"""
Shared output-folder naming helpers for DROCAT.

Every main function creates one top-level, timestamped run folder under the
user's output directory, named:

    {tool}_{dataset_abbreviation}_{detail}_{timestamp}

Examples:
    findpath_MCNS_aMe12_to_aMe10_L2w3r0p0_20260801_183000
    finddirect_MCNS_aMe12_to_aMe10_L2w3r0p0_20260801_183005
    profiling_MCNS_aMe12_aMe10_aMe9_20260801_183010
    findhomologs_MCNS_to_HEMI_aMe12_20260801_183015
    findlines_MCNS_aMe12_20260801_183020
    plot3d_MCNS_aMe12_20260801_183025
"""


DATASET_ABBREVIATIONS = {
    "male-cns": "MCNS",
    "male_cns": "MCNS",
    "hemibrain": "HEMI",
    "optic-lobe": "OL",
    "optic_lobe": "OL",
    "manc": "MANC",
    "banc": "BANC",
    "fib19": "FIB",
    "mushroombody": "MB",
    "flywire_fafb": "FAFB",
    "fafb": "FAFB",
    "flywire_banc": "BANC",
}


def dataset_abbrev(dataset) -> str:
    """Return a short, folder-safe abbreviation for a dataset identifier."""
    if not dataset:
        return "UNKN"
    ds = str(dataset).lower()
    for key, abbrev in DATASET_ABBREVIATIONS.items():
        if key in ds:
            return abbrev
    letters = "".join(c for c in ds.split(":")[0] if c.isalpha())
    return (letters[:4] or "DS").upper()

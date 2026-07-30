"""Raw-data loaders shared between the kentucky/ and us/ pipelines.

Each function is a direct extraction of code that used to be copy-pasted,
with small variations, across both projects' notebooks -- behavior is
preserved exactly; parameters cover the variation (footnote-stripping,
index suffixes, dtype casting) that existed between call sites so nothing
downstream changes silently.
"""

import pandas as pd


def load_puma_migpuma(
    path: str,
) -> pd.DataFrame:
    """Load and index the PUMA<->MIGPUMA equivalency csv."""
    puma_migpuma = pd.read_csv(path, dtype="str").set_index("PUMA")
    return puma_migpuma[puma_migpuma.index.notna()]


def load_distance_matrix(
    path: str,
    index_dtype: type | str | None = None,
    columns_dtype: type | str | None = None,
    zfill: int | None = None,
) -> pd.DataFrame:
    """Load a PUMA/MIGPUMA distance (or time) matrix CSV."""
    distances = pd.read_csv(path, index_col=0)
    if zfill:
        distances.index = distances.index.astype(str).str.zfill(zfill)
        distances.columns = distances.columns.map(str).str.zfill(zfill)
    else:
        if index_dtype is not None:
            distances.index = distances.index.astype(index_dtype)
        if columns_dtype is not None:
            distances.columns = distances.columns.astype(columns_dtype)
    return distances

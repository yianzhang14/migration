from collections.abc import Callable

import numpy as np
import pandas as pd

from .model_spec import (
    MOVE_ONLY_SPECS,
    MOVE_ONLY_TERMS,
    SHARED_SPECS,
    SHARED_TERMS,
    STAY_ONLY_SPECS,
    STAY_ONLY_TERMS,
    DestOnlySpec,
    SharedSpec,
)

NUM_PUMAS = 2336

def build_long_data(
    df_train: pd.DataFrame,
    num_alternatives: int,
    dtype: np.dtype | type = np.float32,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Reshape `df_train` (as returned by `lib.io.read_estdata`) into the long `(person_id, alt)`
    table  used by larch/torch choice

    `MOVE_ONLY_TERMS` are structurally zero at `alt=0` (they only make sense as a move decision, e.g.
    `destchoice_samestate`) and vice versa.

    `log_pop_offset` is destination population (`ALT{i}_TOT_POP`) on move rows and origin population
    (`Total Population....ORIG`) on the stay row -- an un-parameterized (coefficient == 1, not
    estimated) term

    Returns `(long_df, STAY_ONLY_TERMS, SHARED_TERMS, MOVE_ONLY_TERMS)`. `long_df` is sorted by
    `(person_id, alt)`, so each person occupies a contiguous run of `num_alternatives + 1` rows in a
    fixed alt order.
    """
    n = len(df_train)
    num_alts = num_alternatives + 1

    # Sorting `df_train` itself (or `long_df` at the end) would copy a frame that is
    # num_alternatives-wide; instead permute each `(n,)` column as it is read.
    person_id = df_train["person_id"].to_numpy()
    order = None
    if not np.all(person_id[:-1] <= person_id[1:]):
        order = np.argsort(person_id, kind="stable")
        person_id = person_id[order]

    def col(name: str) -> np.ndarray:
        values = df_train[name].to_numpy()
        return values if order is None else values[order]

    feature_names = (
        ["log_pop_offset"] + STAY_ONLY_TERMS + SHARED_TERMS + MOVE_ONLY_TERMS
    )
    # Fortran order so that each column is a contiguous run of the buffer and can be reshaped
    # into an (n, num_alts) *view*: writes below land directly in their final position, with
    # row `i * num_alts + alt` already in (person_id, alt) order. Zero-initialized, so the
    # structural zeros need no fill pass.
    data = np.zeros((n * num_alts, len(feature_names)), dtype=dtype, order="F")
    block = {
        name: data[:, j].reshape(n, num_alts) for j, name in enumerate(feature_names)
    }

    # fixed (coefficient == 1, not estimated) log-population offset -- origin population on the
    # stay row, destination population on the move rows.
    pop = block["log_pop_offset"]
    # staying does not have a size term, there is not multiple ways to stay
    pop[:, 0] = 0
    for i in range(1, num_alts):
        pop[:, i] = np.log(col(f"ALT{i}_TOT_POP"))

    # stay-only terms (structurally zero at alt>=1)
    for spec in STAY_ONLY_SPECS:
        value = 1.0 if spec.source is None else col(spec.source)
        scale = spec.scale(col)
        block[spec.name][:, 0] = value if scale is None else value * scale
    origin_type = col("TYPE_NUM.ORIG")
    block["stay_T34"][:, 0] = origin_type == 0
    block["stay_metro"][:, 0] = origin_type == 1

    # shared terms
    for spec in SHARED_SPECS:
        scale = spec.scale(col)
        target = block[spec.name]
        values = col(spec.origin_col)
        target[:, 0] = values if scale is None else values * scale
        for i in range(1, num_alts):
            values = col(f"ALT{i}_{spec.alt_suffix}")
            target[:, i] = values if scale is None else values * scale

    # destination-only terms (structurally zero at alt=0, like the hand-written move-only terms
    # below, but read generically from ALT{i}_<suffix> the same way SHARED_SPECS is)
    for spec in MOVE_ONLY_SPECS:
        target = block[spec.name]
        for i in range(1, num_alts):
            scale = spec.scale(col, i)
            values = col(f"ALT{i}_{spec.alt_suffix}")
            target[:, i] = values if scale is None else values * scale

    # move-only terms (structurally zero at alt=0, since staying isn't a move-type decision)
    origin_state = col("ORIGIN_STATE")
    origin_cbsa = col("NAME_NUM.ORIG")
    birth_state = col("POBP")
    # `origin_type_t34 * alt_type_t34 + origin_type_metro * alt_type_metro + ...` is an equality
    # test over the three coded types spelled out as a sum of products; NaN/unknown codes fall
    # through both forms as 0.
    origin_type_coded = (origin_type >= 0) & (origin_type <= 2)

    logdist = block["destchoice_logdist"]
    samecbsa = block["destchoice_samecbsa"]
    samestate = block["destchoice_samestate"]
    birthstate = block["destchoice_birthstate"]
    dest_t34 = block["destchoice_T34"]
    dest_metro = block["destchoice_metro"]
    same_type = block["destchoice_same_cbsa_type"]

    for i in range(1, num_alts):
        alt_state = col(f"ALT{i}_STATE")
        alt_type = col(f"ALT{i}_TYPE")
        logdist[:, i] = np.log1p(col(f"ALT{i}_DIST"))
        samecbsa[:, i] = origin_cbsa == col(f"ALT{i}_CBSA")
        samestate[:, i] = origin_state == alt_state
        birthstate[:, i] = birth_state == alt_state
        dest_t34[:, i] = alt_type == 0
        dest_metro[:, i] = alt_type == 1
        # effective or between similar types
        same_type[:, i] = (origin_type == alt_type) & origin_type_coded

    # The shared terms and the offset are required non-null in *both* contexts; a NaN there is a
    # data problem, not a structural absence, and would silently become a 0.0 below.
    missing = [
        name
        for name in SHARED_TERMS + ["log_pop_offset"]
        if np.isnan(block[name]).any()
    ]
    assert not missing, missing

    # ...whereas a NaN in a stay-only/move-only source is filled with 0.0, matching the blanket
    # `fillna(0.0)` this replaces (structural zeros are already zero from the allocation).
    for name in STAY_ONLY_TERMS:
        stay_values = block[name][:, 0]
        np.copyto(stay_values, 0.0, where=np.isnan(stay_values))
    for name in MOVE_ONLY_TERMS:
        move_values = block[name][:, 1:]
        np.copyto(move_values, 0.0, where=np.isnan(move_values))

    # create long df, combining the stay, destination, and shared variables
    long_df = pd.DataFrame(data, columns=feature_names, copy=False)

    # insert choice model details: choice is 1 if the alternative was chosen
    # alt says which alterantive a line represents
    # person id tells which person a line's alternative pertains to
    alt = np.tile(np.arange(num_alts), n)
    long_df.insert(
        0, "choice", (alt == np.repeat(col("ALT_CHOICE"), num_alts)).astype(int)
    )
    long_df.insert(0, "alt", alt)
    long_df.insert(0, "person_id", np.repeat(person_id, num_alts))

    long_df["sampling_correction"] = np.log(NUM_PUMAS / num_alternatives)
    # NOTE: this assumes that staying is alternative 0
    long_df["sampling_correction"] = np.where(long_df["alt"] == 0, 0, long_df["sampling_correction"])

    # sanity checks
    num_persons = df_train["person_id"].nunique()
    assert num_persons == n, (num_persons, n)
    assert len(long_df) == num_persons * num_alts, (
        len(long_df),
        num_persons * num_alts,
    )

    return long_df, list(STAY_ONLY_TERMS), list(SHARED_TERMS), list(MOVE_ONLY_TERMS)


def print_utility_formula(long_df: pd.DataFrame) -> None:
    """Print the stay and destination utility functions implied by `long_df` (as returned by
    build_long_data). Each term is classified stay-only/move-only/shared by checking where
    it's structurally zero in the data, not by importing the spec lists."""
    stay_rows = long_df["alt"] == 0
    move_rows = ~stay_rows
    terms = [c for c in long_df.columns if c not in ("person_id", "alt", "choice")]

    stay_terms, move_terms, shared_terms = [], [], []
    for t in terms:
        zero_at_stay = (long_df.loc[stay_rows, t] == 0).all()
        zero_at_move = (long_df.loc[move_rows, t] == 0).all()
        if zero_at_move and not zero_at_stay:
            stay_terms.append(t)
        elif zero_at_stay and not zero_at_move:
            move_terms.append(t)
        else:
            shared_terms.append(t)

    shared_by_name = {spec.name: spec for spec in SHARED_SPECS}
    move_only_specs = {spec.name: spec for spec in MOVE_ONLY_SPECS}

    def factors_str(spec: SharedSpec | DestOnlySpec) -> str:
        return "".join(
            f"*Variable({f})" if isinstance(f, str) else f"*{f}" for f in spec.factors
        )

    def stay_term(t: str) -> str:
        spec = shared_by_name.get(t)
        if spec is None:
            return f"Beta({t})*Variable({t})"
        return f"Beta({t})*Variable({spec.origin_col}){factors_str(spec)}"

    def move_term(t: str) -> str:
        spec = shared_by_name.get(t) or move_only_specs.get(t)
        if spec is None:
            return f"Beta({t})*Variable({t})"
        return f"Beta({t})*Variable(ALT{{i}}_{spec.alt_suffix}){factors_str(spec)}"

    def formula(terms: list[str], term_fn: Callable[[str], str]) -> str:
        lines = [term_fn(t) for t in terms]
        return "    " + "\n    + ".join(lines)

    print("Stay utility:")
    print(formula(stay_terms + shared_terms, stay_term))
    print("Destination utility:")
    print(formula(move_terms + shared_terms, move_term))

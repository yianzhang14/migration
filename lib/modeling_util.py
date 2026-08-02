from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .globals import ALT_VARYING_SUFFIXES, INDIV_COLS

# non-spec columns build_long_data reads directly (log-pop-offset, move-only terms)
_STRUCTURAL_INDIVIDUAL_COLS = {
    "Total Population.Total Population.SE_A00001_001.ORIG",
    "TYPE_NUM.ORIG",
    "ORIGIN_STATE",
    "NAME_NUM.ORIG",
    "POBP",
}
_STRUCTURAL_ALT_SUFFIXES = {"TOT_POP", "DIST", "CBSA", "STATE", "TYPE"}


@dataclass(frozen=True)
class StayOnlySpec:
    """One term in `_STAY_ONLY_SPECS`: `name`'s value on the stay row is `1.0` if `source` is
    None (this is how `stay`, the alternative-specific constant for staying, is defined), else
    `df_train[source]`."""

    name: str
    source: str | None = None


_STAY_ONLY_SPECS: list[StayOnlySpec] = [
    StayOnlySpec("stay"),
    StayOnlySpec("stay_age_18_22", "AGE_18_22"),
    StayOnlySpec("stay_age_23_29", "AGE_23_29"),
    StayOnlySpec("stay_age_30_39", "AGE_30_39"),
    StayOnlySpec("stay_age_40_49", "AGE_40_49"),
    StayOnlySpec("stay_age_50_64", "AGE_50_64"),
    StayOnlySpec("stay_child_under_6", "CHILD_UNDER_6"),
    StayOnlySpec("stay_child_6_to_17", "CHILD_6_TO_17"),
    StayOnlySpec("stay_married_more_than_year", "MARRIED_MORE_THAN_YEAR"),
    StayOnlySpec("stay_married_less_than_year", "RECENTLY_MARRIED"),
    StayOnlySpec("stay_recently_divorced_or_widowed", "RECENTLY_WIDOWED_OR_DIVORCED"),
    StayOnlySpec("stay_2work_mar", "WORK2_MAR"),
    StayOnlySpec("stay_single_parent", "SINGLE_PARENT"),
    StayOnlySpec("stay_edu_college", "EDU_BACHELORS"),
    StayOnlySpec("stay_edu_high", "EDU_HIGH"),
    StayOnlySpec("stay_in_college", "IN_COLLEGE"),
    StayOnlySpec("stay_foreign", "FOREIGN"),
    # NOTE: this assumes that NAICS code stays constant between the origin and destination
    StayOnlySpec("stay_mil", "IN_MILITARY"),
    StayOnlySpec("stay_naics_govt", "NAICS_GOVT"),
    StayOnlySpec("stay_naics_goods_trade", "NAICS_GOODS_TRADE"),
    StayOnlySpec("stay_naics_license", "NAICS_LICENSE"),
    StayOnlySpec("stay_naics_high_ed", "NAICS_HIGH_ED"),
    StayOnlySpec("stay_naics_agr_ext", "NAICS_AGR_EXT"),
]

STAY_ONLY_TERMS = [spec.name for spec in _STAY_ONLY_SPECS] + ["stay_T34", "stay_metro"]


@dataclass(frozen=True)
class SharedSpec:
    """One term shared between the stay and move contexts: `name`'s value is `origin_col`
    (person vs. their origin) on the stay row and `ALT{i}_{alt_suffix}` (mover vs. destination)
    on move rows, both scaled by the product of `factors` -- each factor is either a
    `df_train` column name or a plain scalar."""

    name: str
    origin_col: str
    alt_suffix: str
    factors: tuple[str | float, ...] = ()

    def scale(self, col: Callable[[str], np.ndarray]) -> np.ndarray | float | None:
        if not self.factors:
            return None
        result: np.ndarray | float = 1.0
        for factor in self.factors:
            result = result * (col(factor) if isinstance(factor, str) else factor)
        return result


_SHARED_SPECS: list[SharedSpec] = [
    SharedSpec(
        "proportion_same_age_18_34",
        "Proportion of people 18-34.ORIG",
        "OWN_AGE_PROP",
        ("AGE_18_34",),
    ),
    SharedSpec(
        "proportion_same_age_35_64",
        "Proportion of people 35-64.ORIG",
        "OWN_AGE_PROP",
        ("AGE_35_64",),
    ),
    SharedSpec(
        "proportion_same_age_65_plus",
        "Proportion of people 65+.ORIG",
        "OWN_AGE_PROP",
        ("AGE_OVER_65",),
    ),
    SharedSpec(
        "proportion_hh_with_children_if_have_children",
        "Proportion of households with children.ORIG",
        "HH_WITH_CHILD_PROP",
        ("CHILD",),
    ),
    SharedSpec(
        "proportion_college_if_in_college",
        "Proportion of people in college.ORIG",
        "COLLEGE_PROP",
        ("IN_COLLEGE",),
    ),
    SharedSpec(
        "proportion_foreign_if_foreign",
        "Proportion foreign born.ORIG",
        "FOREIGN_BORN_PROP",
        ("FOREIGN",),
    ),
    SharedSpec(
        "median_hh_income_in_tens_of_thousands",
        "Median Household Income (In 2018 Inflation Adjusted Dollars).Median Household Income (In 2018 Inflation Adjusted Dollars).SE_A14006_001.ORIG",
        "HH_MED_INC",
        (1 / 10_000,),
    ),
    SharedSpec(
        "median_house_value_over_median_income",
        "Median house value over median household income.ORIG",
        "MED_HOUSE_VALUE_OVER_MED_HH_INC",
    ),
    SharedSpec(
        "median_gross_rent_percentage_hh_inc",
        "Median gross rent as a percentage of household income.ORIG",
        "MED_RENT_PROP_HH_INC",
    ),
    SharedSpec("vacancy_rate", "House vacancy proportion.ORIG", "HOUSE_VACANCY_PROP"),
    SharedSpec("median_travel_time", "Median travel time.ORIG", "MED_TRAVEL_TIME"),
    SharedSpec(
        "proportion_ent", "Proportion of entertainment jobs.ORIG", "ENT_JOBS_PROP"
    ),
    SharedSpec(
        "proportion_ent_18_34",
        "Proportion of entertainment jobs.ORIG",
        "ENT_JOBS_PROP",
        ("AGE_18_34",),
    ),
    SharedSpec(
        "proportion_ent_35_64",
        "Proportion of entertainment jobs.ORIG",
        "ENT_JOBS_PROP",
        ("AGE_35_64",),
    ),
    SharedSpec(
        "proportion_also_mil",
        "Proportion of people in military.ORIG",
        "MIL_PROP",
        ("IN_MILITARY",),
    ),
    SharedSpec(
        "proportion_same_naics_govt",
        "NAICS_GROUP_PROP_GOVT.ORIG",
        "OWN_NAICS_GROUP_PROP",
        ("NAICS_GOVT",),
    ),
    SharedSpec(
        "proportion_same_naics_goods_trade",
        "NAICS_GROUP_PROP_GOODS_TRADE.ORIG",
        "OWN_NAICS_GROUP_PROP",
        ("NAICS_GOODS_TRADE",),
    ),
    SharedSpec(
        "proportion_same_naics_license",
        "NAICS_GROUP_PROP_LICENSE.ORIG",
        "OWN_NAICS_GROUP_PROP",
        ("NAICS_LICENSE",),
    ),
    SharedSpec(
        "proportion_same_naics_high_ed",
        "NAICS_GROUP_PROP_HIGH_ED.ORIG",
        "OWN_NAICS_GROUP_PROP",
        ("NAICS_HIGH_ED",),
    ),
    SharedSpec(
        "proportion_same_naics_agr_ext",
        "NAICS_GROUP_PROP_AGR_EXT.ORIG",
        "OWN_NAICS_GROUP_PROP",
        ("NAICS_AGR_EXT",),
    ),
    SharedSpec(
        "proportion_same_race_black",
        "Proportion of people Black.ORIG",
        "OWN_RACE_ETH_PROP",
        ("BLACK",),
    ),
    SharedSpec(
        "proportion_same_race_aapi",
        "Proportion of people AAPI.ORIG",
        "OWN_RACE_ETH_PROP",
        ("AAPI",),
    ),
    SharedSpec(
        "proportion_same_race_indian",
        "Proportion of people Indian.ORIG",
        "OWN_RACE_ETH_PROP",
        ("INDIAN",),
    ),
    SharedSpec(
        "proportion_also_latino",
        "Proportion of people Latino.ORIG",
        "OWN_RACE_ETH_PROP",
        ("LATINO",),
    ),
    SharedSpec(
        "proportion_same_race_white",
        "Proportion of people White.ORIG",
        "OWN_RACE_ETH_PROP",
        ("WHITE",),
    ),
]

SHARED_TERMS = [spec.name for spec in _SHARED_SPECS]

MOVE_ONLY_TERMS = [
    "destchoice_logdist",
    "destchoice_samecbsa",
    "destchoice_samestate",
    "destchoice_birthstate",
    "destchoice_T34",
    "destchoice_metro",
    "destchoice_same_cbsa_type",
]


def required_alt_suffixes() -> set[str]:
    """`ALT{i}_<suffix>` suffixes build_long_data reads: structural ones plus each SharedSpec's alt_suffix."""
    return _STRUCTURAL_ALT_SUFFIXES | {spec.alt_suffix for spec in _SHARED_SPECS}


def required_individual_columns() -> set[str]:
    """Plain df_train columns build_long_data reads: StayOnlySpec sources, SharedSpec origin_col/factors, plus structural ones."""
    cols = set(_STRUCTURAL_INDIVIDUAL_COLS)
    cols.update(spec.source for spec in _STAY_ONLY_SPECS if spec.source is not None)
    for spec in _SHARED_SPECS:
        cols.add(spec.origin_col)
        cols.update(factor for factor in spec.factors if isinstance(factor, str))
    return cols


# catches a spec referencing a column read_estdata won't have pulled in, before it's a runtime KeyError
_missing_alt_suffixes = required_alt_suffixes() - set(ALT_VARYING_SUFFIXES)
assert not _missing_alt_suffixes, _missing_alt_suffixes
_missing_individual_cols = required_individual_columns() - set(INDIV_COLS)
assert not _missing_individual_cols, _missing_individual_cols


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
    pop[:, 0] = np.log(col("Total Population.Total Population.SE_A00001_001.ORIG"))
    for i in range(1, num_alts):
        pop[:, i] = np.log(col(f"ALT{i}_TOT_POP"))

    # stay-only terms (structurally zero at alt>=1)
    for spec in _STAY_ONLY_SPECS:
        block[spec.name][:, 0] = 1.0 if spec.source is None else col(spec.source)
    origin_type = col("TYPE_NUM.ORIG")
    block["stay_T34"][:, 0] = origin_type == 0
    block["stay_metro"][:, 0] = origin_type == 1

    # shared terms
    for spec in _SHARED_SPECS:
        scale = spec.scale(col)
        target = block[spec.name]
        values = col(spec.origin_col)
        target[:, 0] = values if scale is None else values * scale
        for i in range(1, num_alts):
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

    shared_by_name = {spec.name: spec for spec in _SHARED_SPECS}

    def factors_str(spec: SharedSpec) -> str:
        return "".join(
            f"*Variable({f})" if isinstance(f, str) else f"*{f}" for f in spec.factors
        )

    def stay_term(t: str) -> str:
        spec = shared_by_name.get(t)
        if spec is None:
            return f"Beta({t})*Variable({t})"
        return f"Beta({t})*Variable({spec.origin_col}){factors_str(spec)}"

    def move_term(t: str) -> str:
        spec = shared_by_name.get(t)
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

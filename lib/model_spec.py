from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

BASE_INDIVIDUAL_COLS = {
    "TOT_POP.ORIG",
    "TYPE_NUM.ORIG",
    "ORIGIN_STATE",
    "NAME_NUM.ORIG",
    "BPL_REF",
    "CHOSEN",
    "STAY",
    "PERWT",
}
BASE_ALT_SUFFIXES = {"TOT_POP", "DIST", "CBSA", "STATE", "TYPE"}


def _scale(
    factors: Iterable[str | float], col: Callable[[str], np.ndarray]
) -> np.ndarray | float | None:
    if not factors:
        return None
    result: np.ndarray | float = 1.0
    for factor in factors:
        result = result * (col(factor) if isinstance(factor, str) else factor)
    return result


@dataclass(frozen=True)
class StayOnlySpec:
    """One term in `STAY_ONLY_SPECS`: `name`'s value on the stay row is `1.0` if `source` is
    None (this is how `stay`, the alternative-specific constant for staying, is defined), else
    `df_train[source]`, scaled by the product of `factors` (see `SharedSpec`)."""

    name: str
    source: str | None = None
    factors: tuple[str | float, ...] = ()

    def scale(self, col: Callable[[str], np.ndarray]) -> np.ndarray | float | None:
        return _scale(self.factors, col)


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
        return _scale(self.factors, col)


@dataclass(frozen=True)
class DestOnlySpec:
    """`StayOnlySpec`'s mirror image: one term in `DEST_ONLY_SPECS` with no origin-side
    counterpart. `name`'s value is `ALT{i}_{alt_suffix}` (mover vs. destination) on move rows,
    scaled by the product of `factors` (see `SharedSpec`), and structurally `0` on the stay
    row -- so it belongs in `MOVE_ONLY_TERMS` like the hand-written `destchoice_*` terms."""

    name: str
    alt_suffix: str
    factors: tuple[str | float, ...] = ()

    def scale(
        self, col: Callable[[str], np.ndarray], alt: int
    ) -> np.ndarray | float | None:
        factors: list[str | float] = [
            factor.replace("ALT_", f"ALT{alt}_") if isinstance(factor, str) else factor
            for factor in self.factors
        ]
        return _scale(factors, col)


STAY_ONLY_SPECS: list[StayOnlySpec] = [
    StayOnlySpec("stay"),
    StayOnlySpec("stay_age_18_22", "AGE_18_22"),
    StayOnlySpec("stay_age_23_29", "AGE_23_29"),
    StayOnlySpec("stay_age_30_39", "AGE_30_39"),
    StayOnlySpec("stay_age_40_49", "AGE_40_49"),
    StayOnlySpec("stay_age_50_64", "AGE_50_64"),
    StayOnlySpec("stay_child_under_6", "CHILD_UNDER_6"),
    StayOnlySpec("stay_child_6_to_17", "CHILD_6_TO_17"),
    StayOnlySpec("stay_married_more_than_year", "MARRIED_MORE_THAN_YEAR", factors=("PAIRED_UNIT",)),
    StayOnlySpec("stay_married_less_than_year", "RECENTLY_MARRIED", factors=("PAIRED_UNIT",)),
    StayOnlySpec("stay_recently_divorced_or_widowed", "RECENTLY_WIDOWED_OR_DIVORCED"),
    StayOnlySpec("stay_2work", "WORK2"),
    StayOnlySpec("stay_single_parent", "SINGLE_UNIT_WITH_CHILD"),
    StayOnlySpec("stay_edu_at_least_bachelors", "EDU_BACHELORS_OR_HIGHER"),
    StayOnlySpec("stay_edu_high_no_bachelors", "EDU_HIGH_BUT_NOT_BACHELORS"),
    StayOnlySpec("stay_in_college", "IN_COLLEGE"),
    StayOnlySpec("stay_foreign", "FOREIGN_BORN"),
    StayOnlySpec("stay_mil", "IN_MILITARY"),
    StayOnlySpec(
        "stay_med_earnings_10k_no_degree",
        "OWN_EARNINGS_10K_BY_EDU.ORIG",
        ("EDU_NO_DEGREE",),
    ),
    StayOnlySpec(
        "stay_med_earnings_10k_degree",
        "OWN_EARNINGS_10K_BY_EDU.ORIG",
        ("EDU_HAS_DEGREE",),
    ),
    StayOnlySpec("stay_med_rent_k", "Median gross rent in thousands of dollars.ORIG"),
    StayOnlySpec("stay_vacancy_rate", "House vacancy proportion.ORIG"),
    StayOnlySpec("stay_alt_commute", "Proportion alternative commute.ORIG"),
]

STAY_ONLY_TERMS = [spec.name for spec in STAY_ONLY_SPECS] + ["stay_T34", "stay_metro"]


SHARED_SPECS: list[SharedSpec] = [
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
        ("FOREIGN_BORN",),
    ),
    SharedSpec("median_travel_time", "Median travel time.ORIG", "MED_TRAVEL_TIME"),
    SharedSpec(
        "proportion_also_mil",
        "Proportion of people in military.ORIG",
        "MIL_PROP",
        ("IN_MILITARY",),
    ),
    # NOTE: this assumes that NAICS code stays constant between the origin and destination
    # Split by member for the same reason as race: 65% of dual-earner couples work in
    # different skill/credential groups, so an averaged indicator would attribute one
    # partner's job share to the other's group.
    SharedSpec(
        "proportion_same_naics_govt",
        "OWN_NAICS_GROUP_PROP_REF.ORIG",
        "OWN_NAICS_GROUP_PROP_REF",
        ("NAICS_GOVT_REF", "NAICS_W_REF"),
    ),
    SharedSpec(
        "proportion_same_naics_govt",
        "OWN_NAICS_GROUP_PROP_SEC.ORIG",
        "OWN_NAICS_GROUP_PROP_SEC",
        ("NAICS_GOVT_SEC", "NAICS_W_SEC"),
    ),
    SharedSpec(
        "proportion_same_naics_goods_trade",
        "OWN_NAICS_GROUP_PROP_REF.ORIG",
        "OWN_NAICS_GROUP_PROP_REF",
        ("NAICS_GOODS_TRADE_REF", "NAICS_W_REF"),
    ),
    SharedSpec(
        "proportion_same_naics_goods_trade",
        "OWN_NAICS_GROUP_PROP_SEC.ORIG",
        "OWN_NAICS_GROUP_PROP_SEC",
        ("NAICS_GOODS_TRADE_SEC", "NAICS_W_SEC"),
    ),
    SharedSpec(
        "proportion_same_naics_high_ed",
        "OWN_NAICS_GROUP_PROP_REF.ORIG",
        "OWN_NAICS_GROUP_PROP_REF",
        ("NAICS_HIGH_ED_REF", "NAICS_W_REF"),
    ),
    SharedSpec(
        "proportion_same_naics_high_ed",
        "OWN_NAICS_GROUP_PROP_SEC.ORIG",
        "OWN_NAICS_GROUP_PROP_SEC",
        ("NAICS_HIGH_ED_SEC", "NAICS_W_SEC"),
    ),
    SharedSpec(
        "proportion_same_naics_agr_ext",
        "OWN_NAICS_GROUP_PROP_REF.ORIG",
        "OWN_NAICS_GROUP_PROP_REF",
        ("NAICS_AGR_EXT_REF", "NAICS_W_REF"),
    ),
    SharedSpec(
        "proportion_same_naics_agr_ext",
        "OWN_NAICS_GROUP_PROP_SEC.ORIG",
        "OWN_NAICS_GROUP_PROP_SEC",
        ("NAICS_AGR_EXT_SEC", "NAICS_W_SEC"),
    ),
    # Race terms are split by member: OWN_RACE_ETH_PROP_{REF,SEC} is that member's own
    # category share at the location, so pairing it with that same member's indicator
    # isolates their contribution. The two weighted specs share a `name` and are
    # SUMMED by build_long_data, giving one coefficient whose regressor is
    # (unit's fraction in the category) x (area's share of it).
    SharedSpec(
        "proportion_same_race_black",
        "OWN_RACE_ETH_PROP_REF.ORIG",
        "OWN_RACE_ETH_PROP_REF",
        ("BLACK_REF", "RACE_W_REF"),
    ),
    SharedSpec(
        "proportion_same_race_black",
        "OWN_RACE_ETH_PROP_SEC.ORIG",
        "OWN_RACE_ETH_PROP_SEC",
        ("BLACK_SEC", "RACE_W_SEC"),
    ),
    SharedSpec(
        "proportion_same_race_aapi",
        "OWN_RACE_ETH_PROP_REF.ORIG",
        "OWN_RACE_ETH_PROP_REF",
        ("AAPI_REF", "RACE_W_REF"),
    ),
    SharedSpec(
        "proportion_same_race_aapi",
        "OWN_RACE_ETH_PROP_SEC.ORIG",
        "OWN_RACE_ETH_PROP_SEC",
        ("AAPI_SEC", "RACE_W_SEC"),
    ),
    SharedSpec(
        "proportion_same_race_indian",
        "OWN_RACE_ETH_PROP_REF.ORIG",
        "OWN_RACE_ETH_PROP_REF",
        ("INDIAN_REF", "RACE_W_REF"),
    ),
    SharedSpec(
        "proportion_same_race_indian",
        "OWN_RACE_ETH_PROP_SEC.ORIG",
        "OWN_RACE_ETH_PROP_SEC",
        ("INDIAN_SEC", "RACE_W_SEC"),
    ),
    SharedSpec(
        "proportion_also_latino",
        "OWN_RACE_ETH_PROP_REF.ORIG",
        "OWN_RACE_ETH_PROP_REF",
        ("LATINO_REF", "RACE_W_REF"),
    ),
    SharedSpec(
        "proportion_also_latino",
        "OWN_RACE_ETH_PROP_SEC.ORIG",
        "OWN_RACE_ETH_PROP_SEC",
        ("LATINO_SEC", "RACE_W_SEC"),
    ),
    SharedSpec(
        "proportion_same_race_white",
        "OWN_RACE_ETH_PROP_REF.ORIG",
        "OWN_RACE_ETH_PROP_REF",
        ("WHITE_REF", "RACE_W_REF"),
    ),
    SharedSpec(
        "proportion_same_race_white",
        "OWN_RACE_ETH_PROP_SEC.ORIG",
        "OWN_RACE_ETH_PROP_SEC",
        ("WHITE_SEC", "RACE_W_SEC"),
    ),
    SharedSpec("jan_avg_temp_c", "JAN_AVG_TEMP_C.ORIG", "JAN_AVG_TEMP_C"),
    # SharedSpec("rainfall_m", "AVG_TOT_PPT_M.ORIG", "AVG_TOT_PPT_M"),
    SharedSpec(
        "proportion_ent_jobs", "Proportion of entertainment jobs.ORIG", "ENT_JOB_PROP"
    ),
    SharedSpec(
        "proportion_ent_jobs_18_34",
        "Proportion of entertainment jobs.ORIG",
        "ENT_JOB_PROP",
        ("AGE_18_34",),
    ),
    SharedSpec(
        "proportion_ent_jobs_35_64",
        "Proportion of entertainment jobs.ORIG",
        "ENT_JOB_PROP",
        ("AGE_35_64",),
    ),
]


# de-duplicated: race/NAICS categories each contribute two specs (one per member) that
# share a name and are summed by build_long_data, so they remain ONE coefficient.
SHARED_TERMS = list(dict.fromkeys(spec.name for spec in SHARED_SPECS))

# purely destination choice terms that can be automatically handled
MOVE_ONLY_SPECS: list[DestOnlySpec] = [
    DestOnlySpec("destchoice_vacancy_rate", "HOUSE_VACANCY_PROP"),
    DestOnlySpec("destchoice_med_rent_k", "MED_RENT_K"),
    DestOnlySpec(
        "destchoice_med_earnings_10k_no_degree",
        "OWN_EARNINGS_10K_BY_EDU",
        ("EDU_NO_DEGREE",),
    ),
    DestOnlySpec(
        "destchoice_med_earnings_10k_degree",
        "OWN_EARNINGS_10K_BY_EDU",
        ("EDU_HAS_DEGREE",),
    ),
    DestOnlySpec("destchoice_unemp", "UNEMP_RATE"),
    DestOnlySpec("destchoice_alt_commute", "ALT_COMMUTE_PROP"),
]

# the ones not defined by a spec are computed ad-hoc in modeling_util.py
MOVE_ONLY_TERMS = [
    "destchoice_logdist",
    "destchoice_samecbsa",
    "destchoice_samestate",
    "destchoice_birthstate",
    "destchoice_T34",
    "destchoice_metro",
    "destchoice_same_cbsa_type",
] + [spec.name for spec in MOVE_ONLY_SPECS]


def required_alt_suffixes() -> set[str]:
    """`ALT{i}_<suffix>` suffixes build_long_data reads: structural ones plus each SharedSpec's
    and DestOnlySpec's alt_suffix."""
    return (
        BASE_ALT_SUFFIXES
        | {spec.alt_suffix for spec in SHARED_SPECS}
        | {spec.alt_suffix for spec in MOVE_ONLY_SPECS}
    )


def required_individual_columns() -> set[str]:
    """Plain df_train columns build_long_data reads: StayOnlySpec source/factors, SharedSpec
    origin_col/factors, DestOnlySpec factors, plus structural ones."""
    cols = set(BASE_INDIVIDUAL_COLS)
    for spec in STAY_ONLY_SPECS:
        if spec.source is not None:
            cols.add(spec.source)
        cols.update(factor for factor in spec.factors if isinstance(factor, str))
    for spec in SHARED_SPECS:
        cols.add(spec.origin_col)
        cols.update(factor for factor in spec.factors if isinstance(factor, str))
    for spec in MOVE_ONLY_SPECS:
        cols.update(factor for factor in spec.factors if isinstance(factor, str))
    return cols

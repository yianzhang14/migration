from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# non-spec columns build_long_data reads directly (log-pop-offset, move-only terms)
STRUCTURAL_INDIVIDUAL_COLS = {
    "Total Population.Total Population.SE_A00001_001.ORIG",
    "TYPE_NUM.ORIG",
    "ORIGIN_STATE",
    "NAME_NUM.ORIG",
    "POBP",
    "CHOSEN",
    "STAY"
}
STRUCTURAL_ALT_SUFFIXES = {"TOT_POP", "DIST", "CBSA", "STATE", "TYPE"}


@dataclass(frozen=True)
class StayOnlySpec:
    """One term in `_STAY_ONLY_SPECS`: `name`'s value on the stay row is `1.0` if `source` is
    None (this is how `stay`, the alternative-specific constant for staying, is defined), else
    `df_train[source]`."""

    name: str
    source: str | None = None


STAY_ONLY_SPECS: list[StayOnlySpec] = [
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
    # StayOnlySpec("stay_naics_goods_trade", "NAICS_GOODS_TRADE"),
    # StayOnlySpec("stay_naics_license", "NAICS_LICENSE"),
    # StayOnlySpec("stay_naics_high_ed", "NAICS_HIGH_ED"),
    # StayOnlySpec("stay_naics_agr_ext", "NAICS_AGR_EXT"),
]

STAY_ONLY_TERMS = [spec.name for spec in STAY_ONLY_SPECS] + ["stay_T34", "stay_metro"]


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
        ("FOREIGN",),
    ),
    SharedSpec(
        "med_earnings_10k",
        "Median earnings in thousands of dollars.ORIG",
        "MED_EARNINGS_K",
        (0.1, )
    ),
    # SharedSpec(
    #     "med_house_val_100k",
    #     "Median house cost in hundreds of thousands of dollars.ORIG",
    #     "MED_HOUSE_VAL_100k"
    # ),
    SharedSpec(
        "med_rent_1k",
        "Median gross rent in thousands of dollars.ORIG",
        "MED_RENT_K"
    ),
    # SharedSpec(
    #     "median_house_value_over_median_income",
    #     "Median house value over median household income.ORIG",
    #     "MED_HOUSE_VALUE_OVER_MED_HH_INC",
    # ),
    # SharedSpec(
    #     "median_gross_rent_percentage_hh_inc",
    #     "Median gross rent as a percentage of household income.ORIG",
    #     "MED_RENT_PROP_HH_INC",
    # ),
    SharedSpec("vacancy_rate", "House vacancy proportion.ORIG", "HOUSE_VACANCY_PROP"),
    SharedSpec("median_travel_time", "Median travel time.ORIG", "MED_TRAVEL_TIME"),
    SharedSpec(
        "amenities_est_per_capita", "AMENITIES_EST_PER_CAPITA.ORIG", "AMENITIES_EST_PER_CAPITA"
    ),
    SharedSpec(
        "amenities_est_per_capita_18_34",
        "AMENITIES_EST_PER_CAPITA.ORIG",
        "AMENITIES_EST_PER_CAPITA",
        ("AGE_18_34",),
    ),
    SharedSpec(
        "amenities_est_per_capita_35_64",
        "AMENITIES_EST_PER_CAPITA.ORIG",
        "AMENITIES_EST_PER_CAPITA",
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
    SharedSpec(
        "jan_avg_temp_c",
        "JAN_AVG_TEMP_C.ORIG",
        "JAN_AVG_TEMP_C"
    ),
    SharedSpec(
        "lf_prop_if_in_lf",
        "Labor force participation rate.ORIG",
        "LF_PARTCP_RATE",
    )
]

SHARED_TERMS = [spec.name for spec in SHARED_SPECS]

# these are computed ad-hoc in modeling_util.py
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
    return STRUCTURAL_ALT_SUFFIXES | {spec.alt_suffix for spec in SHARED_SPECS}


def required_individual_columns() -> set[str]:
    """Plain df_train columns build_long_data reads: StayOnlySpec sources, SharedSpec origin_col/factors, plus structural ones."""
    cols = set(STRUCTURAL_INDIVIDUAL_COLS)
    cols.update(spec.source for spec in STAY_ONLY_SPECS if spec.source is not None)
    for spec in SHARED_SPECS:
        cols.add(spec.origin_col)
        cols.update(factor for factor in spec.factors if isinstance(factor, str))
    return cols

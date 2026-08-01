import numpy as np
import pandas as pd

from .io import ALT_VARYING_SUFFIXES


def build_long_data(
    df_train: pd.DataFrame,
    num_alternatives: int,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Reshape `df_train` (as returned by `lib.io.read_estdata`) into the long `(person_id, alt)`
    table shared by the torch-choice and Larch ports of `modeling_mnl.ipynb`: `alt=0` is staying,
    `alt=1..num_alternatives` are the move alternatives.

    `MOVE_ONLY_TERMS` are structurally zero at `alt=0` (they only make sense as a move decision, e.g.
    `destchoice_samestate`). `STAY_ONLY_TERMS` are structurally zero at `alt=1..num_alternatives`
    (e.g. `stay`, the alternative-specific constant for staying). `SHARED_TERMS` hold the move-context
    value (comparing the mover to the destination) on move rows and the stay-context value (comparing
    the person to their origin) on the `alt=0` row -- both are populated under the same column name,
    which is what ties them to a single shared coefficient downstream (in Biogeme terms, both contexts
    use the same named `Beta`).

    `log_pop_offset` is destination population (`ALT{i}_TOT_POP`) on move rows and origin population
    (`Total Population....ORIG`) on the stay row -- an un-parameterized (coefficient == 1, not
    estimated) term in both this notebook's source model and Biogeme's un-wrapped `log(Variable(...))`
    calls.

    Returns `(long_df, STAY_ONLY_TERMS, SHARED_TERMS, MOVE_ONLY_TERMS)`. `long_df` is sorted by
    `(person_id, alt)`, so each person occupies a contiguous run of `num_alternatives + 1` rows in a
    fixed alt order.
    """
    move_long = wide_to_long(
        df_train,
        alt_list=[f"ALT{i}" for i in range(1, num_alternatives + 1)],
        alt_name="alt_label",
        varying=ALT_VARYING_SUFFIXES,
        sep="_",
        alt_is_prefix=True,
    )
    move_long["alt"] = move_long["alt_label"].str[len("ALT") :].astype(int)
    move_long["choice"] = (move_long["alt"] == move_long["ALT_CHOICE"]).astype(int)

    same_state = (move_long["ORIGIN_STATE"] == move_long["STATE"]).astype(float)
    # NAME_NUM.ORIG and ALT{i}_CBSA are factorized against the same CBSA-name codebook
    # (see create_estdata.ipynb), so they're directly comparable
    same_cbsa = (move_long["NAME_NUM.ORIG"] == move_long["CBSA"]).astype(float)
    origin_type_t34 = (move_long["TYPE_NUM.ORIG"] == 0).astype(float)
    origin_type_metro = (move_long["TYPE_NUM.ORIG"] == 1).astype(float)
    origin_type_nonmetro = (move_long["TYPE_NUM.ORIG"] == 2).astype(float)
    alt_type_t34 = (move_long["TYPE"] == 0).astype(float)
    alt_type_metro = (move_long["TYPE"] == 1).astype(float)
    alt_type_nonmetro = (move_long["TYPE"] == 2).astype(float)

    # fixed (coefficient == 1, not estimated) log-population offset -- destination population for move rows.
    move_long["log_pop_offset"] = np.log(move_long["TOT_POP"])

    # move-only terms (structurally zero at alt=0, since staying isn't a move-type decision)
    move_long["destchoice_logdist"] = np.log(move_long["DIST"] + 1)
    move_long["destchoice_samecbsa"] = same_cbsa
    move_long["destchoice_samestate"] = same_state
    move_long["destchoice_birthstate"] = (
        move_long["POBP"] == move_long["STATE"]
    ).astype(float)
    # origin area type x destination area type; nonmetro_nonmetro is the omitted reference category,
    # matching the commented-out c_destchoice_nonmetro_nonmetro in modeling_mnl.ipynb's cell 22.
    move_long["destchoice_T34"] = alt_type_t34
    move_long["destchoice_metro"] = alt_type_metro
    # effective or between similar types
    move_long["destchoice_same_cbsa_type"] = (
        origin_type_t34 * alt_type_t34
        + origin_type_metro * alt_type_metro
        + origin_type_nonmetro * alt_type_nonmetro
    )

    # shared terms (move-context value: comparing the mover to the destination)
    move_long["proportion_same_age_18_34"] = (
        move_long["AGE_18_34"] * move_long["OWN_AGE_PROP"]
    )
    move_long["proportion_same_age_35_64"] = (
        move_long["AGE_35_64"] * move_long["OWN_AGE_PROP"]
    )
    move_long["proportion_same_age_65_plus"] = (
        move_long["AGE_OVER_65"] * move_long["OWN_AGE_PROP"]
    )
    move_long["proportion_hh_with_children_if_have_children"] = (
        move_long["HH_WITH_CHILD_PROP"] * move_long["CHILD"]
    )
    move_long["proportion_college_if_in_college"] = (
        move_long["IN_COLLEGE"] * move_long["COLLEGE_PROP"]
    )
    move_long["proportion_foreign_if_foreign"] = (
        move_long["FOREIGN"] * move_long["FOREIGN_BORN_PROP"]
    )
    move_long["median_hh_income_in_tens_of_thousands"] = (
        move_long["HH_MED_INC"] / 10_000
    )
    move_long["median_house_value_over_median_income"] = move_long[
        "MED_HOUSE_VALUE_OVER_MED_HH_INC"
    ]
    move_long["median_gross_rent_percentage_hh_inc"] = move_long["MED_RENT_PROP_HH_INC"]
    move_long["unemp_rate"] = move_long["UNEMP_RATE"]
    move_long["vacancy_rate"] = move_long["HOUSE_VACANCY_PROP"]
    move_long["median_travel_time"] = move_long["MED_TRAVEL_TIME"]
    move_long["proportion_alt_commute"] = move_long["ALT_COMMUTE_PROP"]
    move_long["proportion_ent"] = move_long["ENT_JOBS_PROP"]
    move_long["proportion_ent_18_34"] = (
        move_long["AGE_18_34"] * move_long["ENT_JOBS_PROP"]
    )
    move_long["proportion_ent_35_64"] = (
        move_long["AGE_35_64"] * move_long["ENT_JOBS_PROP"]
    )
    move_long["proportion_also_mil"] = move_long["IN_MILITARY"] * move_long["MIL_PROP"]
    # c_proportion_same_naics_goods_trade uses ALT{i}_OWN_NAICS_GROUP_PROP here, not the mangled
    # f"{alt}OWN_GROUPOWN_NAICS_GROUP_PROP_PROP" that modeling_mnl.ipynb's cell 23 currently has.
    move_long["proportion_same_naics_govt"] = (
        move_long["NAICS_GOVT"] * move_long["OWN_NAICS_GROUP_PROP"]
    )
    move_long["proportion_same_naics_goods_trade"] = (
        move_long["NAICS_GOODS_TRADE"] * move_long["OWN_NAICS_GROUP_PROP"]
    )
    move_long["proportion_same_naics_license"] = (
        move_long["NAICS_LICENSE"] * move_long["OWN_NAICS_GROUP_PROP"]
    )
    move_long["proportion_same_naics_high_ed"] = (
        move_long["NAICS_HIGH_ED"] * move_long["OWN_NAICS_GROUP_PROP"]
    )
    move_long["proportion_same_naics_agr_ext"] = (
        move_long["NAICS_AGR_EXT"] * move_long["OWN_NAICS_GROUP_PROP"]
    )
    move_long["proportion_same_race_black"] = (
        move_long["BLACK"] * move_long["OWN_RACE_ETH_PROP"]
    )
    move_long["proportion_same_race_aapi"] = (
        move_long["AAPI"] * move_long["OWN_RACE_ETH_PROP"]
    )
    move_long["proportion_same_race_indian"] = (
        move_long["INDIAN"] * move_long["OWN_RACE_ETH_PROP"]
    )
    move_long["proportion_also_latino"] = (
        move_long["LATINO"] * move_long["OWN_RACE_ETH_PROP"]
    )
    move_long["proportion_same_race_white"] = (
        move_long["WHITE"] * move_long["OWN_RACE_ETH_PROP"]
    )

    MOVE_ONLY_TERMS = [
        "destchoice_logdist",
        "destchoice_samecbsa",
        "destchoice_samestate",
        "destchoice_birthstate",
        "destchoice_T34_T34",
        "destchoice_T34_metro",
        "destchoice_T34_nonmetro",
        "destchoice_metro_T34",
        "destchoice_metro_metro",
        "destchoice_metro_nonmetro",
        "destchoice_nonmetro_T34",
        "destchoice_nonmetro_metro",
    ]
    SHARED_TERMS = [
        "proportion_same_age_18_34",
        "proportion_same_age_35_64",
        "proportion_same_age_65_plus",
        "proportion_hh_with_children_if_have_children",
        "proportion_college_if_in_college",
        "proportion_foreign_if_foreign",
        "median_hh_income_in_tens_of_thousands",
        "median_house_value_over_median_income",
        "median_gross_rent_percentage_hh_inc",
        "unemp_rate",
        "vacancy_rate",
        "median_travel_time",
        "proportion_alt_commute",
        "proportion_ent",
        "proportion_ent_18_34",
        "proportion_ent_35_64",
        "proportion_also_mil",
        "proportion_same_naics_govt",
        "proportion_same_naics_goods_trade",
        "proportion_same_naics_license",
        "proportion_same_naics_high_ed",
        "proportion_same_naics_agr_ext",
        "proportion_same_race_black",
        "proportion_same_race_aapi",
        "proportion_same_race_indian",
        "proportion_also_latino",
        "proportion_same_race_white",
    ]
    move_long = move_long[
        ["person_id", "alt", "choice", "log_pop_offset"]
        + MOVE_ONLY_TERMS
        + SHARED_TERMS
    ]

    stay = df_train.copy()
    stay["stay"] = 1.0  # c_stay: the stay alternative-specific constant
    stay["stay_age_18_22"] = stay["AGE_18_22"]
    stay["stay_age_23_29"] = stay["AGE_23_29"]
    stay["stay_age_30_39"] = stay["AGE_30_39"]
    stay["stay_age_40_49"] = stay["AGE_40_49"]
    stay["stay_age_50_64"] = stay["AGE_50_64"]

    stay["stay_child_under_6"] = stay["CHILD_UNDER_6"]
    stay["stay_child_6_to_17"] = stay["CHILD_6_TO_17"]

    stay["stay_married_more_than_year"] = stay["MARRIED_MORE_THAN_YEAR"]
    stay["stay_married_less_than_year"] = stay["RECENTLY_MARRIED"]
    stay["stay_recently_divorced_or_widowed"] = stay["RECENTLY_WIDOWED_OR_DIVORCED"]
    stay["stay_2work_mar"] = stay["WORK2_MAR"]
    stay["stay_single_parent"] = stay["SINGLE_PARENT"]

    stay["stay_edu_college"] = stay["EDU_BACHELORS"]
    stay["stay_edu_high"] = stay["EDU_HIGH"]

    stay["stay_in_college"] = stay["IN_COLLEGE"]
    stay["stay_foreign"] = stay["FOREIGN"]

    # NOTE: this assumes that NAICS code stays constant between the origin and destination
    stay["stay_mil"] = stay["IN_MILITARY"]
    stay["stay_naics_govt"] = stay["NAICS_GOVT"]
    stay["stay_naics_goods_trade"] = stay["NAICS_GOODS_TRADE"]
    stay["stay_naics_license"] = stay["NAICS_LICENSE"]
    stay["stay_naics_high_ed"] = stay["NAICS_HIGH_ED"]
    stay["stay_naics_agr_ext"] = stay["NAICS_AGR_EXT"]

    stay["stay_T34"] = origin_type_t34
    stay["stay_metro"] = origin_type_metro

    STAY_ONLY_TERMS = [
        "stay",
        "stay_age_18_22",
        "stay_age_23_29",
        "stay_age_30_39",
        "stay_age_40_49",
        "stay_age_50_64",
        "stay_child_under_6",
        "stay_child_6_to_17",
        "stay_married_more_than_year",
        "stay_married_less_than_year",
        "stay_recently_divorced_or_widowed",
        "stay_2work_mar",
        "stay_single_parent",
        "stay_edu_college",
        "stay_edu_high",
        "stay_in_college",
        "stay_foreign",
        "stay_mil",
        "stay_naics_govt",
        "stay_naics_goods_trade",
        "stay_naics_license",
        "stay_naics_high_ed",
        "stay_naics_agr_ext",
        "stay_T34",
        "stay_metro",
    ]

    HH_INCOME_COL = (
        "Median Household Income (In 2018 Inflation Adjusted Dollars)."
        "Median Household Income (In 2018 Inflation Adjusted Dollars).SE_A14006_001.ORIG"
    )

    # fixed (coefficient == 1, not estimated) log-population offset -- origin population for the stay row.
    stay["log_pop_offset"] = np.log(
        stay["Total Population.Total Population.SE_A00001_001.ORIG"]
    )

    # shared terms (stay-context value: comparing the person to their origin)
    stay["proportion_same_age_18_34"] = (
        stay["Proportion of people 18-34.ORIG"] * stay["AGE_18_34"]
    )
    stay["proportion_same_age_35_64"] = (
        stay["Proportion of people 35-64.ORIG"] * stay["AGE_35_64"]
    )
    stay["proportion_same_age_65_plus"] = (
        stay["Proportion of people 65+.ORIG"] * stay["AGE_OVER_65"]
    )
    stay["proportion_hh_with_children_if_have_children"] = (
        stay["Proportion of households with children.ORIG"] * stay["CHILD"]
    )
    stay["proportion_college_if_in_college"] = (
        stay["Proportion of people in college.ORIG"] * stay["IN_COLLEGE"]
    )
    stay["proportion_foreign_if_foreign"] = (
        stay["Proportion foreign born.ORIG"] * stay["FOREIGN"]
    )
    stay["median_hh_income_in_tens_of_thousands"] = stay[HH_INCOME_COL] / 10_000
    stay["median_house_value_over_median_income"] = stay[
        "Median house value over median household income.ORIG"
    ]
    stay["median_gross_rent_percentage_hh_inc"] = stay[
        "Median gross rent as a percentage of household income.ORIG"
    ]
    stay["unemp_rate"] = stay["Unemployment rate.ORIG"]
    stay["vacancy_rate"] = stay["House vacancy proportion.ORIG"]
    stay["median_travel_time"] = stay["Median travel time.ORIG"]
    stay["proportion_alt_commute"] = stay["Proportion alternative commute.ORIG"]
    stay["proportion_ent"] = stay["Proportion of entertainment jobs.ORIG"]
    stay["proportion_ent_18_34"] = (
        stay["AGE_18_34"] * stay["Proportion of entertainment jobs.ORIG"]
    )
    stay["proportion_ent_35_64"] = (
        stay["AGE_35_64"] * stay["Proportion of entertainment jobs.ORIG"]
    )
    stay["proportion_also_mil"] = (
        stay["Proportion of people in military.ORIG"] * stay["IN_MILITARY"]
    )
    stay["proportion_same_naics_govt"] = (
        stay["NAICS_GROUP_PROP_GOVT.ORIG"] * stay["NAICS_GOVT"]
    )
    stay["proportion_same_naics_goods_trade"] = (
        stay["NAICS_GROUP_PROP_GOODS_TRADE.ORIG"] * stay["NAICS_GOODS_TRADE"]
    )
    stay["proportion_same_naics_license"] = (
        stay["NAICS_GROUP_PROP_LICENSE.ORIG"] * stay["NAICS_LICENSE"]
    )
    stay["proportion_same_naics_high_ed"] = (
        stay["NAICS_GROUP_PROP_HIGH_ED.ORIG"] * stay["NAICS_HIGH_ED"]
    )
    stay["proportion_same_naics_agr_ext"] = (
        stay["NAICS_GROUP_PROP_AGR_EXT.ORIG"] * stay["NAICS_AGR_EXT"]
    )
    stay["proportion_same_race_black"] = (
        stay["Proportion of people Black.ORIG"] * stay["BLACK"]
    )
    stay["proportion_same_race_aapi"] = (
        stay["Proportion of people AAPI.ORIG"] * stay["AAPI"]
    )
    stay["proportion_same_race_indian"] = (
        stay["Proportion of people Indian.ORIG"] * stay["INDIAN"]
    )
    stay["proportion_also_latino"] = (
        stay["Proportion of people Latino.ORIG"] * stay["LATINO"]
    )
    stay["proportion_same_race_white"] = (
        stay["Proportion of people White.ORIG"] * stay["WHITE"]
    )

    stay["alt"] = 0
    stay["choice"] = (stay["ALT_CHOICE"] == 0).astype(int)
    stay = stay[
        ["person_id", "alt", "choice", "log_pop_offset"]
        + STAY_ONLY_TERMS
        + SHARED_TERMS
    ]

    long_df = pd.concat([stay, move_long], ignore_index=True, sort=False)
    varnames = STAY_ONLY_TERMS + SHARED_TERMS + MOVE_ONLY_TERMS
    long_df[varnames + ["log_pop_offset"]] = long_df[
        varnames + ["log_pop_offset"]
    ].fillna(0.0)
    long_df = long_df.sort_values(["person_id", "alt"]).reset_index(drop=True)

    num_persons = df_train["person_id"].nunique()
    num_alts = num_alternatives + 1
    assert len(long_df) == num_persons * num_alts, (
        len(long_df),
        num_persons * num_alts,
    )

    return long_df, STAY_ONLY_TERMS, SHARED_TERMS, MOVE_ONLY_TERMS


def wide_to_long(
    df: pd.DataFrame,
    alt_list: list[str],
    alt_name: str,
    varying: list[str],
    sep: str,
    alt_is_prefix: bool,
) -> pd.DataFrame:
    assert alt_is_prefix and sep == "_"
    varying_cols = {f"{alt}{sep}{suf}" for alt in alt_list for suf in varying}
    id_vars = [c for c in df.columns if c not in varying_cols]
    frames = []
    for alt in alt_list:
        rename = {f"{alt}{sep}{suf}": suf for suf in varying}
        sub = df[id_vars + list(rename.keys())].rename(columns=rename)
        sub[alt_name] = alt
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)

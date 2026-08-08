"""Per-geography (PUMA / MIGPUMA) census tables and the column codebooks that go with them.

Extracted from `data/create_estdata.ipynb` so that anything scoring the model outside of
that notebook (e.g. `us/simulation.ipynb`) reads the destination-side data through exactly
the same joins, CBSA encoding and derived columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

UNIT_SUFFIXES = ["REF", "SEC"]

# curated subset of ACS/LODES fields: `ALT{i}_<key>` <- `<value>` in the per-geography table
CENSUS_FIELDS = {
    "TOT_POP": "TOT_POP",
    "FOREIGN_BORN_PROP": "Proportion foreign born",
    "MIL_PROP": "Proportion of people in military",
    "MED_HOUSE_VAL_100k": "Median house cost in hundreds of thousands of dollars",
    "MED_RENT_K": "Median gross rent in thousands of dollars",
    "UNEMP_RATE": "Unemployment rate",
    "COLLEGE_PROP": "Proportion of people in college",
    "HOUSE_VACANCY_PROP": "House vacancy proportion",
    "MED_TRAVEL_TIME": "Median travel time",
    "HH_WITH_CHILD_PROP": "Proportion of households with children",
    "JAN_AVG_TEMP_C": "JAN_AVG_TEMP_C",
    "AVG_TOT_PPT_M": "AVG_TOT_PPT_M",
    "ALT_COMMUTE_PROP": "Proportion alternative commute",  # walking or public transit
    "ENT_JOB_PROP": "Proportion of entertainment jobs",
}

NAICS_SECTOR_COLUMNS = {
    "AGR": "Number of jobs in NAICS sector 11 (Agriculture, Forestry, Fishing and Hunting)",
    "EXT": "Number of jobs in NAICS sector 21 (Mining, Quarrying, and Oil and Gas Extraction)",
    "UTL": "Number of jobs in NAICS sector 22 (Utilities)",
    "CON": "Number of jobs in NAICS sector 23 (Construction)",
    "MFG": "Number of jobs in NAICS sector 31-33 (Manufacturing)",
    "WHL": "Number of jobs in NAICS sector 42 (Wholesale Trade)",
    "RET": "Number of jobs in NAICS sector 44-45 (Retail Trade)",
    "TRN": "Number of jobs in NAICS sector 48-49 (Transportation and Warehousing)",
    "INF": "Number of jobs in NAICS sector 51 (Information)",
    "FIN": "Number of jobs in NAICS sector 52 (Finance and Insurance)",
    "REL": "Number of jobs in NAICS sector 53 (Real Estate and Rental and Leasing)",
    "PRF": "Number of jobs in NAICS sector 54 (Professional, Scientific, and Technical Services)",
    "MNG": "Number of jobs in NAICS sector 55 (Management of Companies and Enterprises)",
    "ADM": "Number of jobs in NAICS sector 56 (Administrative and Support and Waste Management and Remediation Services)",
    "EDU": "Number of jobs in NAICS sector 61 (Educational Services)",
    "MED": "Number of jobs in NAICS sector 62 (Health Care and Social Assistance)",
    "ENT": "Number of jobs in NAICS sector 71 (Arts, Entertainment, and Recreation)",
    "FOD": "Number of jobs in NAICS sector 72 (Accommodation and Food Services)",
    "SRV": "Number of jobs in NAICS sector 81 (Other Services [except Public Administration])",
    "PUB": "Number of jobs in NAICS sector 92 (Public Administration)",
}

# NAICS sectors grouped by skill/credential barrier to entry
NAICS_GROUPS = {
    "AGR_EXT": ["AGR", "EXT"],
    "HIGH_ED": ["MED", "EDU", "PRF", "FIN", "INF", "MNG"],
    "LICENSE": ["SRV", "REL"],
    "GOODS_TRADE": ["MFG", "CON", "WHL", "TRN", "UTL"],
    "LOW_SKILL_SVC": ["RET", "FOD", "ADM", "ENT"],
    "GOVT": ["PUB"],
}
NAICS_GROUP_PROP_COLUMNS = [f"NAICS_GROUP_PROP_{group}" for group in NAICS_GROUPS]

# ACS's NAICSP codes are prefixed by their 2-digit sector ("3M"/"4M" are the merged
# "not specified" manufacturing/retail codes); "99" is other/unemployed.
NAICS_SECTOR_PREFIXES = {
    "11": "AGR",
    "21": "EXT",
    "22": "UTL",
    "23": "CON",
    "31": "MFG",
    "32": "MFG",
    "33": "MFG",
    "3M": "MFG",
    "42": "WHL",
    "44": "RET",
    "45": "RET",
    "4M": "RET",
    "48": "TRN",
    "49": "TRN",
    "51": "INF",
    "52": "FIN",
    "53": "REL",
    "54": "PRF",
    "55": "MNG",
    "56": "ADM",
    "61": "EDU",
    "62": "MED",
    "71": "ENT",
    "72": "FOD",
    "81": "SRV",
    "92": "PUB",
}

AGE_BRACKET_COLS = [
    "Proportion of people under 18",
    "Proportion of people 18-34",
    "Proportion of people 35-64",
    "Proportion of people 65+",
]
AGE_BRACKET_FLAGS = ["AGE_UNDER_18", "AGE_18_34", "AGE_35_64", "AGE_OVER_65"]

# keyed on IPUMS RACE (with Hispanic overriding race as code 99), *not* ACS RAC1P --
# the two agree only on codes 1 and 2.
RACE_COLUMNS = {
    1: "Proportion of people White",
    2: "Proportion of people Black",
    3: "Proportion of people Indian",
    4: "Proportion of people AAPI",
    5: "Proportion of people AAPI",
    6: "Proportion of people AAPI",
    7: "Proportion of people other race",
    8: "Proportion of people other race",
    9: "Proportion of people other race",
    99: "Proportion of people Latino",
}
RACE_CATEGORIES = ["WHITE", "BLACK", "INDIAN", "AAPI", "OTHER_RACE", "LATINO"]

EDU_EARNINGS_RAW_COLS = {
    "EDU_NOHIGH": "Median earnings in tens of thousands of dollars for people without high school diplomas",
    "EDU_ONLY_HIGH": "Median earnings in tens of thousands of dollars for high school graduates only",
    "EDU_SOME_COLLEGE": "Median earnings in tens of thousands of dollars for people with some college only",
    "EDU_ONLY_BACHELORS": "Median earnings in tens of thousands of dollars for people with bachelors only",
    "EDU_GRADUATE_DEG": "Median earnings in tens of thousands of dollars for people with graduate degrees",
}


def encode_cbsa_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add `TYPE_NUM`: T34 -> 0, Metro -> 1, Nonmetro -> 2, anything else -> 3."""
    out = np.full(len(df), 3)
    out = np.where(df["TYPE"] == "Nonmetro", 2, out)
    out = np.where(df["TYPE"] == "Metro", 1, out)
    out = np.where(df["TYPE"] == "T34", 0, out)
    df["TYPE_NUM"] = out
    return df


def add_naics_group_prop_columns(census_df: pd.DataFrame) -> pd.DataFrame:
    """Add each geography's share of all jobs falling in each `NAICS_GROUPS` group."""
    total_jobs = census_df["Total number of jobs"]
    for group, sectors in NAICS_GROUPS.items():
        group_jobs = census_df[[NAICS_SECTOR_COLUMNS[s] for s in sectors]].sum(axis=1)
        census_df[f"NAICS_GROUP_PROP_{group}"] = np.where(
            total_jobs == 0, 0, group_jobs / total_jobs
        )
    return census_df


def _read_indexed(path: Path, key: str, suffix: str = "") -> pd.DataFrame:
    df = pd.read_csv(path, dtype={key: str})
    df[key] = df[key].str.zfill(7)
    df = df.set_index(key)
    return df.add_suffix(suffix) if suffix else df


def load_geo_tables(
    data_dir: str | Path, year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the `(puma_data, migpuma_data)` tables: ACS + LODES + CBSA + weather + CBP
    establishment counts, joined per geography, with derived type/name codes and job shares.

    CBSA names are factorized on the PUMA side and mapped onto the MIGPUMA side (missing ->
    -2), so `NAME_NUM` is comparable across the two -- destinations are PUMAs while origins
    are MIGPUMAs, and the same-CBSA term compares the two directly.
    """
    data_dir = Path(data_dir)

    puma = _read_indexed(data_dir / f"acs/acs_puma_{year}.csv", "PUMA")
    migpuma = _read_indexed(data_dir / f"acs/acs_migpuma_{year}.csv", "MIGPUMA")

    tables = {
        "PUMA": [
            _read_indexed(data_dir / f"lodes/wac_puma_{year}.csv", "PUMA"),
            _read_indexed(data_dir / "weather/puma_weather.csv", "PUMA"),
            _read_indexed(data_dir / f"cbp/puma_est_{year}.csv", "PUMA", "_NUM_EST"),
        ],
        "MIGPUMA": [
            _read_indexed(data_dir / f"lodes/wac_migpuma_{year}.csv", "MIGPUMA"),
            _read_indexed(data_dir / "weather/migpuma_weather.csv", "MIGPUMA"),
            _read_indexed(
                data_dir / f"cbp/migpuma_est_{year}.csv", "MIGPUMA", "_NUM_EST"
            ),
        ],
    }

    puma_cbsa = encode_cbsa_type(
        _read_indexed(data_dir / f"geometry/cbsa/puma_density_{year}.csv", "GEOID")
    )
    migpuma_cbsa = encode_cbsa_type(
        _read_indexed(data_dir / f"geometry/cbsa/migpuma_density_{year}.csv", "GISMATCH")
    )
    puma_cbsa["NAME_NUM"], cbsa_names = pd.factorize(puma_cbsa["CBSA_NAME"])
    migpuma_cbsa["NAME_NUM"] = (
        migpuma_cbsa["CBSA_NAME"]
        .map(dict(zip(cbsa_names, range(len(cbsa_names)))))
        .fillna(-2)
        .astype(int)
    )

    cbsa_cols = ["CBSA_NAME", "NAME_NUM", "TYPE", "TYPE_NUM"]
    out = []
    for base, key, cbsa in (
        (puma, "PUMA", puma_cbsa),
        (migpuma, "MIGPUMA", migpuma_cbsa),
    ):
        data = base
        for table in tables[key]:
            data = data.join(table, how="left")
        data = data.join(cbsa[cbsa_cols].rename_axis(key), how="left")
        data["ENT_EST_PER_CAPITA"] = data["ENT_NUM_EST"] / data["TOT_POP"]
        data["FOD_EST_PER_CAPITA"] = data["FOD_NUM_EST"] / data["TOT_POP"]
        data["AMENITIES_EST_PER_1K_PEOPLE"] = (
            data["ENT_EST_PER_CAPITA"] + data["FOD_EST_PER_CAPITA"]
        ) * 1_000
        data["JOBS_PER_CAPITA"] = data["Total number of jobs"] / data["TOT_POP"]
        out.append(add_naics_group_prop_columns(data))

    return out[0], out[1]

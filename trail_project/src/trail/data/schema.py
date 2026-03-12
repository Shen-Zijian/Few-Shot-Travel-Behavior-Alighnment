"""Unified data schema for harmonized TCS records."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UnifiedTripRecord:
    """
    A single harmonized trip record joining household, person, and trip info.

    All categorical codes are the unified codes defined in harmonization.yaml.
    Missing values use -1 (integer fields) or None (float fields).
    """

    # ---- Identifiers ----
    case_id: str                    # "{survey_year}_{hh_id}_{mem_id}_{trip_no}"
    household_id: str
    person_id: str
    trip_no: int
    survey_year: int                # 2011 or 2022

    # ---- Household attributes ----
    hh_size: int                    # number of household members (-1 = unknown)
    car_availability: int           # 0=no car, 1=1 car, 2=2+ cars

    # ---- Person attributes ----
    age_group: int                  # 1=0-14, 2=15-24, 3=25-44, 4=45-64, 5=65+
    sex: int                        # 1=Male, 2=Female
    employment_status: int          # 1=FT, 2=PT, 3=Self-emp, 4=Student, 5=Homemaker, 6=Retired, 7=Other
    income_group: int               # 0=unknown, 1=low, 2=lower-mid, 3=mid, 4=upper-mid, 5=high

    # ---- Trip context ----
    trip_purpose: int               # 1=Work, 2=Education, 7=Other (see harmonization.yaml)
    departure_period: int           # 1=AM Peak, 2=Inter-peak, 3=PM Peak, 4=Off-peak
    journey_time: Optional[float]   # minutes
    origin_zone: str                # zone ID as string
    destination_zone: str

    # ---- Label ----
    # Shared cross-year 9-class mechanised mode hierarchy:
    # 1=Rail, 2=LRT, 3=Tram, 4=Ferry, 5=PLB, 6=Bus,
    # 7=Private Vehicle, 8=Taxi, 9=SPB
    main_mode: int

    # ---- Weights ----
    trip_weight: float              # survey expansion weight

    # ---- Raw metadata (for debugging) ----
    raw_mode_2011: Optional[str] = None   # original PC/MC/PV combo
    raw_purpose_2011: Optional[str] = None


# Column names in the unified DataFrame (matches dataclass fields)
UNIFIED_COLUMNS = [
    "case_id", "household_id", "person_id", "trip_no", "survey_year",
    "hh_size", "car_availability",
    "age_group", "sex", "employment_status", "income_group",
    "trip_purpose", "departure_period", "journey_time", "origin_zone", "destination_zone",
    "main_mode", "trip_weight",
    "raw_mode_2011", "raw_purpose_2011",
]

# Feature columns used in modeling (excludes identifiers, weights, raw fields)
FEATURE_COLUMNS = [
    "age_group", "sex", "employment_status", "income_group",
    "car_availability", "hh_size",
    "trip_purpose", "departure_period", "journey_time",
]

# Categorical feature columns
CATEGORICAL_COLUMNS = [
    "age_group", "sex", "employment_status", "income_group",
    "car_availability", "trip_purpose", "departure_period",
]

# Numeric feature columns
NUMERIC_COLUMNS = ["journey_time", "hh_size"]

TARGET_COLUMN = "main_mode"

MODE_LABELS = {
    1: "Rail",
    2: "LRT",
    3: "Tram",
    4: "Ferry",
    5: "PLB",
    6: "Bus",
    7: "Private Vehicle",
    8: "Taxi",
    9: "SPB",
}
PURPOSE_LABELS = {1: "Work", 2: "Education", 7: "Other"}
AGE_LABELS = {1: "0-14", 2: "15-24", 3: "25-44", 4: "45-64", 5: "65+"}
SEX_LABELS = {1: "Male", 2: "Female"}
EMPLOYMENT_LABELS = {1: "Full-time", 2: "Part-time", 3: "Self-emp", 4: "Student", 5: "Homemaker", 6: "Retired", 7: "Other"}
INCOME_LABELS = {0: "Unknown", 1: "Low", 2: "Lower-mid", 3: "Middle", 4: "Upper-mid", 5: "High"}
PERIOD_LABELS = {1: "AM Peak", 2: "Inter-peak", 3: "PM Peak", 4: "Off-peak"}
CAR_LABELS = {0: "No car", 1: "1 car", 2: "2+ cars"}

"""Declarative registry of public datasets available to the pipeline."""

from __future__ import annotations

from .models import DatasetSpec, Task

DATASETS: dict[str, DatasetSpec] = {
    "sarcos": DatasetSpec(
        name="sarcos",
        title="SARCOS",
        task="regression",
        source_url="https://data.openml.org/datasets/0004/43873/dataset_43873.pq",
        source_filename="openml_43873.parquet",
        repository_url="https://www.openml.org/d/43873",
        rows=44_484,
        features=21,
        outputs=7,
        split_kind="blocked",
        citation=(
            "Vijayakumar, S. and Schaal, S. (2000). Locally Weighted Projection "
            "Regression: An O(n) Algorithm for Incremental Real Time Learning "
            "in High Dimensional Space. ICML."
        ),
        license="OpenML metadata: Public",
    ),
    "wave_energy": DatasetSpec(
        name="wave_energy",
        title="Wave Energy Converters",
        task="regression",
        source_url=(
            "https://archive.ics.uci.edu/static/public/494/wave+energy+converters.zip"
        ),
        source_filename="uci_494.zip",
        repository_url=(
            "https://archive.ics.uci.edu/dataset/494/wave+energy+converters"
        ),
        rows=287_999,
        features=33,
        outputs=16,
        split_kind="stratified_group",
        citation=(
            "Neshat, M., Wagner, M., and Alexander, B. (2018). Wave Energy "
            "Converters. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5831S"
        ),
        license="CC BY 4.0",
    ),
    "large_wave_energy": DatasetSpec(
        name="large_wave_energy",
        title="Large-scale Wave Energy Farm (49 WECs)",
        task="regression",
        source_url=(
            "https://archive.ics.uci.edu/static/public/882/"
            "large-scale+wave+energy+farm.zip"
        ),
        source_filename="uci_882.zip",
        repository_url=(
            "https://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm"
        ),
        rows=54_007,
        features=99,
        outputs=49,
        split_kind="stratified_group",
        citation=(
            "Neshat, M., Alexander, B., Sergiienko, N., and Wagner, M. (2020). "
            "Large-scale Wave Energy Farm. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5GG7Q"
        ),
        license="CC BY 4.0",
    ),
    "tetouan_power": DatasetSpec(
        name="tetouan_power",
        title="Power Consumption of Tetouan City",
        task="regression",
        source_url=(
            "https://archive.ics.uci.edu/static/public/849/"
            "power+consumption+of+tetouan+city.zip"
        ),
        source_filename="uci_849.zip",
        repository_url=(
            "https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city"
        ),
        rows=52_416,
        features=9,
        outputs=3,
        split_kind="blocked",
        citation=(
            "Salam, A. and El Hibaoui, A. (2018). Power Consumption of Tetouan "
            "City. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5B034"
        ),
        license="CC BY 4.0",
    ),
    "sgemm": DatasetSpec(
        name="sgemm",
        title="SGEMM GPU Kernel Performance",
        task="regression",
        source_url=(
            "https://archive.ics.uci.edu/static/public/440/"
            "sgemm+gpu+kernel+performance.zip"
        ),
        source_filename="uci_440.zip",
        repository_url=(
            "https://archive.ics.uci.edu/dataset/440/sgemm+gpu+kernel+performance"
        ),
        rows=241_600,
        features=14,
        outputs=4,
        split_kind="random",
        citation=(
            "Paredes, E. and Ballester-Ripoll, R. (2017). SGEMM GPU Kernel "
            "Performance. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5MK70"
        ),
        license="CC BY 4.0",
    ),
    "rf1": DatasetSpec(
        name="rf1",
        title="RF1 River Flow",
        task="regression",
        source_url="https://data.openml.org/datasets/0004/41483/dataset_41483.pq",
        source_filename="openml_41483.parquet",
        repository_url="https://www.openml.org/d/41483",
        rows=9_125,
        features=64,
        outputs=8,
        split_kind="purged_blocked",
        citation="Mulan multi-target regression collection; OpenML dataset 41483.",
        license="OpenML metadata: Public",
    ),
    "uji_indoor_loc": DatasetSpec(
        name="uji_indoor_loc",
        title="UJIIndoorLoc",
        task="regression",
        source_url="https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip",
        source_filename="uci_310.zip",
        repository_url="https://archive.ics.uci.edu/dataset/310/ujiindoorloc",
        rows=21_048,
        features=520,
        outputs=2,
        split_kind="official_test",
        citation=(
            "Torres-Sospedra, J. et al. (2014). UJIIndoorLoc. UCI Machine "
            "Learning Repository. https://doi.org/10.24432/C5MS59"
        ),
        license="CC BY 4.0",
    ),
    "covertype": DatasetSpec(
        name="covertype",
        title="Covertype",
        task="classification",
        source_url="https://archive.ics.uci.edu/static/public/31/data.csv",
        source_filename="uci_31.csv",
        repository_url="https://archive.ics.uci.edu/dataset/31/covertype",
        rows=581_012,
        features=54,
        outputs=7,
        split_kind="stratified",
        citation=(
            "Blackard, J. (1998). Covertype. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C50K5N"
        ),
        license="CC BY 4.0",
    ),
    "poker_hand": DatasetSpec(
        name="poker_hand",
        title="Poker Hand",
        task="classification",
        source_url="https://archive.ics.uci.edu/static/public/158/data.csv",
        source_filename="uci_158.csv",
        repository_url="https://archive.ics.uci.edu/dataset/158/poker+hand",
        rows=1_025_010,
        features=10,
        outputs=10,
        split_kind="stratified",
        citation=(
            "Cattral, R. and Oppacher, F. (2002). Poker Hand. UCI Machine "
            "Learning Repository. https://doi.org/10.24432/C5KW38"
        ),
        license="CC BY 4.0",
    ),
    "sensorless_drive": DatasetSpec(
        name="sensorless_drive",
        title="Sensorless Drive Diagnosis",
        task="classification",
        source_url=(
            "https://archive.ics.uci.edu/static/public/325/"
            "dataset+for+sensorless+drive+diagnosis.zip"
        ),
        source_filename="uci_325.zip",
        repository_url=(
            "https://archive.ics.uci.edu/dataset/325/"
            "dataset+for+sensorless+drive+diagnosis"
        ),
        rows=58_509,
        features=48,
        outputs=11,
        split_kind="stratified",
        citation=(
            "Bayer, A., Bator, M., Motsch, J., Bänfer, O., Duda, S., and Enge-"
            "Rosenblatt, O. (2015). Dataset for Sensorless Drive Diagnosis. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5VP5F"
        ),
        license="CC BY 4.0",
    ),
    "letter_recognition": DatasetSpec(
        name="letter_recognition",
        title="Letter Recognition",
        task="classification",
        source_url=(
            "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip"
        ),
        source_filename="uci_59.zip",
        repository_url=("https://archive.ics.uci.edu/dataset/59/letter+recognition"),
        rows=20_000,
        features=16,
        outputs=26,
        split_kind="official_test",
        citation=(
            "Slate, D. (1991). Letter Recognition. UCI Machine Learning "
            "Repository. https://doi.org/10.24432/C5ZP40"
        ),
        license="CC BY 4.0",
    ),
    "devnagari_script": DatasetSpec(
        name="devnagari_script",
        title="Devnagari Script",
        task="classification",
        source_url="https://data.openml.org/datasets/0004/40923/dataset_40923.pq",
        source_filename="openml_40923.parquet",
        repository_url="https://www.openml.org/d/40923",
        rows=92_000,
        features=1_024,
        outputs=46,
        split_kind="stratified",
        citation=(
            "Acharya, S., Pant, A. K., and Gyawali, P. K. (2015). Deep learning "
            "based large scale handwritten Devanagari character recognition."
        ),
        license="OpenML metadata: Public",
    ),
    "gas_sensor_drift": DatasetSpec(
        name="gas_sensor_drift",
        title="Gas Sensor Array Drift",
        task="classification",
        source_url=(
            "https://archive.ics.uci.edu/static/public/224/"
            "gas+sensor+array+drift+dataset.zip"
        ),
        source_filename="uci_224.zip",
        repository_url=(
            "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset"
        ),
        rows=13_910,
        features=128,
        outputs=6,
        split_kind="predefined",
        citation=(
            "Vergara, A. et al. (2012). Gas Sensor Array Drift Dataset. UCI "
            "Machine Learning Repository. https://doi.org/10.24432/C5RP6W"
        ),
        license="CC BY 4.0",
    ),
    "emnist_balanced": DatasetSpec(
        name="emnist_balanced",
        title="EMNIST Balanced",
        task="classification",
        source_url="https://data.openml.org/datasets/0004/41039/dataset_41039.pq",
        source_filename="openml_41039.parquet",
        repository_url="https://www.openml.org/d/41039",
        rows=131_600,
        features=784,
        outputs=47,
        split_kind="stratified",
        citation=(
            "Cohen, G., Afshar, S., Tapson, J., and van Schaik, A. (2017). "
            "EMNIST: an extension of MNIST to handwritten letters."
        ),
        license="OpenML metadata: Public",
    ),
    "kuzushiji_49": DatasetSpec(
        name="kuzushiji_49",
        title="Kuzushiji-49",
        task="classification",
        source_url="https://data.openml.org/datasets/0004/41991/dataset_41991.pq",
        source_filename="openml_41991.parquet",
        repository_url="https://www.openml.org/d/41991",
        rows=270_912,
        features=784,
        outputs=49,
        split_kind="stratified",
        citation=(
            "Clanuwat, T. et al. (2018). Deep Learning for Classical Japanese "
            "Literature. arXiv:1812.01718."
        ),
        license="CC BY-SA 4.0",
    ),
    "dionis": DatasetSpec(
        name="dionis",
        title="Dionis",
        task="classification",
        source_url="https://data.openml.org/datasets/0004/41167/dataset_41167.pq",
        source_filename="openml_41167.parquet",
        repository_url="https://www.openml.org/d/41167",
        rows=416_188,
        features=60,
        outputs=355,
        split_kind="stratified",
        citation="ChaLearn AutoML Challenge, round 3; OpenML dataset 41167.",
        license="OpenML metadata: Public",
    ),
    "aloi": DatasetSpec(
        name="aloi",
        title="ALOI",
        task="classification",
        source_url="https://data.openml.org/datasets/0004/42396/dataset_42396.pq",
        source_filename="openml_42396.parquet",
        repository_url="https://www.openml.org/d/42396",
        rows=108_000,
        features=128,
        outputs=1_000,
        split_kind="stratified",
        citation=(
            "Rocha, A. and Goldenstein, S. (2014). Multiclass from binary: "
            "Expanding one-vs-all, one-vs-one and ECOC-based approaches. IEEE "
            "Transactions on Neural Networks and Learning Systems 25(2)."
        ),
        license="OpenML metadata: Public",
    ),
}


def _openml_url(data_id: int) -> str:
    bucket = data_id // 10_000
    directory = f"{data_id:04d}" if data_id < 10_000 else str(data_id)
    return (
        f"https://data.openml.org/datasets/{bucket:04d}/{directory}/"
        f"dataset_{data_id}.pq"
    )


def _openml_categorical(
    name: str,
    title: str,
    data_id: int,
    target: str,
    task: Task,
    rows: int,
    features: int,
    outputs: int,
    license: str,
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        title=title,
        task=task,
        source_url=_openml_url(data_id),
        source_filename=f"openml_{data_id}.parquet",
        repository_url=f"https://www.openml.org/d/{data_id}",
        rows=rows,
        features=features,
        outputs=outputs,
        split_kind=(
            "blocked"
            if name == "bank_marketing"
            else "stratified"
            if task == "classification"
            else "random"
        ),
        citation=f"OpenML dataset {data_id}: {title}.",
        license=license,
        target=target,
    )


# Public, directly downloadable OpenML datasets from categorical_datasets.md.
# The compact records are: name, title, id, target, task, rows, features,
# outputs, license. Feature counts exclude the target.
OpenMLRecord = tuple[str, str, int, str, Task, int, int, int, str]

_OPENML_CATEGORICAL: list[OpenMLRecord] = [
    # Datasets migrated from gbm_cat_bench.
    (
        "ames_housing",
        "Ames Housing",
        42165,
        "SalePrice",
        "regression",
        1_460,
        80,
        1,
        "OpenML metadata: NA",
    ),
    ("adult", "Adult", 179, "class", "classification", 48_842, 14, 2, "Public"),
    (
        "amazon_employee",
        "Amazon Employee Access",
        4135,
        "target",
        "classification",
        32_769,
        9,
        2,
        "Public",
    ),
    ("airlines", "Airlines", 1169, "Delay", "classification", 539_383, 7, 2, "Public"),
    ("kick", "Kick", 41162, "IsBadBuy", "classification", 72_983, 32, 2, "CC0"),
    # TabArena/OpenML core.
    ("anneal", "Anneal", 46906, "classes", "classification", 898, 38, 5, "CC BY 4.0"),
    (
        "german_credit",
        "German Credit",
        46918,
        "good_or_bad_customer",
        "classification",
        1_000,
        20,
        2,
        "CC BY 4.0",
    ),
    (
        "healthcare_insurance",
        "Healthcare Insurance Expenses",
        46931,
        "charges",
        "regression",
        1_338,
        6,
        1,
        "DbCL 1.0",
    ),
    (
        "website_phishing",
        "Website Phishing",
        46963,
        "WebsiteType",
        "classification",
        1_353,
        9,
        3,
        "CC BY 4.0",
    ),
    (
        "fitness_club",
        "Fitness Club",
        46927,
        "attended",
        "classification",
        1_500,
        6,
        2,
        "Public Domain",
    ),
    ("fiat_500", "Used Fiat 500", 46907, "price", "regression", 1_538, 7, 1, "CC0"),
    ("mic", "MIC", 46980, "LET_IS", "classification", 1_699, 111, 8, "CC BY 4.0"),
    (
        "good_customer",
        "Is This a Good Customer?",
        46938,
        "bad_client_target",
        "classification",
        1_723,
        13,
        2,
        "CC0",
    ),
    (
        "marketing_campaign",
        "Marketing Campaign",
        46940,
        "Response",
        "classification",
        2_240,
        25,
        2,
        "Public",
    ),
    (
        "seismic_bumps",
        "Seismic Bumps",
        46956,
        "HighEnergySeismicBump",
        "classification",
        2_584,
        15,
        2,
        "CC BY 4.0",
    ),
    (
        "splice",
        "Splice",
        46958,
        "SiteType",
        "classification",
        3_190,
        60,
        3,
        "CC BY 4.0",
    ),
    (
        "student_dropout",
        "Students Dropout and Academic Success",
        46960,
        "AcademicOutcome",
        "classification",
        4_424,
        36,
        3,
        "CC BY 4.0",
    ),
    ("churn", "Churn", 46915, "CustomerChurned", "classification", 5_000, 19, 2, "MIT"),
    (
        "naticus_droid",
        "NATICUSdroid",
        46969,
        "Malware",
        "classification",
        7_491,
        86,
        2,
        "CC BY 4.0",
    ),
    (
        "coil2000",
        "COIL2000 Insurance Policies",
        46916,
        "MobileHomePolicy",
        "classification",
        9_822,
        85,
        2,
        "CC BY 4.0",
    ),
    (
        "bank_customer_churn",
        "Bank Customer Churn",
        46911,
        "churn",
        "classification",
        10_000,
        10,
        2,
        "Public",
    ),
    (
        "ecommerce_shipping",
        "E-Commerce Shipping",
        46924,
        "ArrivedLate",
        "classification",
        10_999,
        10,
        2,
        "Public Domain",
    ),
    (
        "online_shoppers",
        "Online Shoppers Intention",
        46947,
        "Revenue",
        "classification",
        12_330,
        17,
        2,
        "CC BY 4.0",
    ),
    (
        "in_vehicle_coupon",
        "In-Vehicle Coupon Recommendation",
        46937,
        "AcceptCoupon",
        "classification",
        12_684,
        24,
        2,
        "CC BY 4.0",
    ),
    (
        "hr_job_change",
        "HR Analytics: Job Change",
        46935,
        "LookingForJobChange",
        "classification",
        19_158,
        12,
        2,
        "CC0",
    ),
    (
        "credit_card_default",
        "Credit Card Clients Default",
        46919,
        "DefaultOnPaymentNextMonth",
        "classification",
        30_000,
        23,
        2,
        "CC BY 4.0",
    ),
    (
        "amazon_employee_tabarena",
        "Amazon Employee Access (TabArena)",
        46905,
        "ResourceApproved",
        "classification",
        32_769,
        9,
        2,
        "Public Domain",
    ),
    (
        "bank_marketing",
        "Bank Marketing",
        46910,
        "SubscribeTermDeposit",
        "classification",
        45_211,
        13,
        2,
        "CC BY 4.0",
    ),
    (
        "food_delivery_time",
        "Food Delivery Time",
        46928,
        "Time_taken(min)",
        "regression",
        45_451,
        9,
        1,
        "DbCL 1.0",
    ),
    (
        "kddcup09_appetency_tabarena",
        "KDD Cup 2009 Appetency (TabArena)",
        46939,
        "appetency",
        "classification",
        50_000,
        212,
        2,
        "Public",
    ),
    ("diamonds", "Diamonds", 46923, "price", "regression", 53_940, 9, 1, "MIT"),
    (
        "diabetes130us",
        "Diabetes130US",
        46922,
        "EarlyReadmission",
        "classification",
        71_518,
        47,
        2,
        "CC BY 4.0",
    ),
    (
        "airline_satisfaction",
        "Airline Satisfaction",
        46920,
        "satisfaction",
        "classification",
        129_880,
        21,
        2,
        "CC0",
    ),
    # Additional high-cardinality OpenML stress datasets.
    (
        "adult_1590",
        "Adult (OpenML 1590)",
        1590,
        "class",
        "classification",
        48_842,
        14,
        2,
        "Public",
    ),
    (
        "census_income_openml",
        "Census-Income-KDD (OpenML)",
        42750,
        "income_50k",
        "classification",
        199_523,
        41,
        2,
        "Public",
    ),
    (
        "cylinder_bands",
        "Cylinder Bands",
        6332,
        "band_type",
        "classification",
        540,
        39,
        2,
        "Public",
    ),
    (
        "dresses_sales",
        "Dresses Sales",
        23381,
        "Class",
        "classification",
        500,
        12,
        2,
        "Public",
    ),
    (
        "kdd_internet_usage",
        "KDD Internet Usage",
        981,
        "Who_Pays_for_Access_Work",
        "classification",
        10_108,
        68,
        2,
        "Public",
    ),
    (
        "kddcup09_appetency",
        "KDDCup09 Appetency",
        1111,
        "APPETENCY",
        "classification",
        50_000,
        230,
        2,
        "Public",
    ),
    (
        "kddcup09_churn",
        "KDDCup09 Churn",
        1112,
        "CHURN",
        "classification",
        50_000,
        230,
        2,
        "Public",
    ),
    (
        "kddcup09_upselling",
        "KDDCup09 Upselling",
        1114,
        "UPSELLING",
        "classification",
        50_000,
        230,
        2,
        "Public",
    ),
    ("kdd98", "KDD98", 42343, "TARGET_B", "classification", 82_318, 477, 2, "Public"),
    ("nomao", "Nomao", 1486, "Class", "classification", 34_465, 118, 2, "Public"),
    (
        "open_payments",
        "Open Payments",
        42738,
        "status",
        "classification",
        73_558,
        5,
        2,
        "CC0",
    ),
    (
        "porto_seguro",
        "Porto Seguro",
        41224,
        "target",
        "classification",
        595_212,
        57,
        2,
        "Public",
    ),
    (
        "sf_police_incidents",
        "SF Police Incidents",
        42344,
        "ViolentCrime",
        "classification",
        538_638,
        6,
        2,
        "PDDL",
    ),
    (
        "speed_dating",
        "Speed Dating",
        40536,
        "match",
        "classification",
        8_378,
        120,
        2,
        "Public",
    ),
    (
        "telco_customer_churn",
        "Telco Customer Churn",
        42178,
        "Churn",
        "classification",
        7_043,
        19,
        2,
        "Public",
    ),
    ("titanic", "Titanic", 40945, "survived", "classification", 1_309, 13, 2, "Public"),
    (
        "wmo_hurricane",
        "WMO Hurricane Survival",
        43607,
        "Class",
        "classification",
        5_021,
        22,
        2,
        "CC0",
    ),
]

for openml_record in _OPENML_CATEGORICAL:
    spec = _openml_categorical(*openml_record)
    DATASETS[spec.name] = spec


def _uci_categorical(
    name: str,
    title: str,
    data_id: int,
    target: str,
    task: Task,
    rows: int,
    features: int,
    outputs: int,
    categorical_features: tuple[str, ...] = (),
    drop_features: tuple[str, ...] = (),
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        title=title,
        task=task,
        source_url=f"https://archive.ics.uci.edu/static/public/{data_id}/data.csv",
        source_filename=f"uci_{data_id}.csv",
        repository_url=f"https://archive.ics.uci.edu/dataset/{data_id}",
        rows=rows,
        features=features,
        outputs=outputs,
        split_kind=(
            "blocked"
            if name == "bike_sharing"
            else "stratified"
            if task == "classification"
            else "random"
        ),
        citation=f"{title}. UCI Machine Learning Repository, dataset {data_id}.",
        license="CC BY 4.0",
        target=target,
        categorical_features=categorical_features,
        drop_features=drop_features,
    )


UCIRecord = tuple[
    str, str, int, str, Task, int, int, int, tuple[str, ...], tuple[str, ...]
]

_UCI_CATEGORICAL: list[UCIRecord] = [
    ("abalone", "Abalone", 1, "Rings", "regression", 4_177, 8, 1, ("Sex",), ()),
    (
        "audiology",
        "Audiology (Standardized)",
        8,
        "class",
        "classification",
        226,
        70,
        24,
        ("*",),
        (),
    ),
    (
        "automobile",
        "Automobile",
        10,
        "price",
        "regression",
        201,
        25,
        1,
        (
            "make",
            "fuel-type",
            "aspiration",
            "num-of-doors",
            "body-style",
            "drive-wheels",
            "engine-location",
            "engine-type",
            "num-of-cylinders",
            "fuel-system",
        ),
        (),
    ),
    (
        "breast_cancer",
        "Breast Cancer Recurrence",
        14,
        "Class",
        "classification",
        286,
        9,
        2,
        ("*",),
        (),
    ),
    (
        "car_evaluation",
        "Car Evaluation",
        19,
        "class",
        "classification",
        1_728,
        6,
        4,
        ("*",),
        (),
    ),
    (
        "chess_kr_vs_kp",
        "Chess KR-vs-KP",
        22,
        "wtoeg",
        "classification",
        3_196,
        35,
        2,
        ("*",),
        (),
    ),
    (
        "connect4",
        "Connect-4",
        26,
        "class",
        "classification",
        67_557,
        42,
        3,
        ("*",),
        (),
    ),
    (
        "credit_approval",
        "Credit Approval",
        27,
        "A16",
        "classification",
        690,
        15,
        2,
        (),
        (),
    ),
    (
        "contraceptive",
        "Contraceptive Method Choice",
        30,
        "contraceptive_method",
        "classification",
        1_473,
        9,
        3,
        (
            "wife_edu",
            "husband_edu",
            "wife_religion",
            "wife_working",
            "husband_occupation",
            "standard_of_living_index",
            "media_exposure",
        ),
        (),
    ),
    (
        "lymphography",
        "Lymphography",
        63,
        "class",
        "classification",
        148,
        19,
        4,
        ("*",),
        (),
    ),
    (
        "monks",
        "MONK's Problems",
        70,
        "class",
        "classification",
        432,
        6,
        2,
        ("*",),
        ("ID",),
    ),
    (
        "mushroom",
        "Mushroom",
        73,
        "poisonous",
        "classification",
        8_124,
        22,
        2,
        ("*",),
        (),
    ),
    (
        "nursery",
        "Nursery",
        76,
        "class",
        "classification",
        12_960,
        8,
        5,
        ("*",),
        (),
    ),
    (
        "tic_tac_toe",
        "Tic-Tac-Toe Endgame",
        101,
        "class",
        "classification",
        958,
        9,
        2,
        ("*",),
        (),
    ),
    (
        "congressional_voting",
        "Congressional Voting Records",
        105,
        "Class",
        "classification",
        435,
        16,
        2,
        ("*",),
        (),
    ),
    (
        "census_income_uci",
        "Census-Income KDD (UCI)",
        117,
        "income",
        "classification",
        299_285,
        41,
        2,
        (),
        (),
    ),
    (
        "german_credit_uci",
        "German Credit (UCI)",
        144,
        "class",
        "classification",
        1_000,
        20,
        2,
        (),
        (),
    ),
    (
        "bike_sharing",
        "Bike Sharing",
        275,
        "cnt",
        "regression",
        17_389,
        12,
        1,
        (
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
        ),
        ("instant", "dteday", "casual", "registered"),
    ),
    (
        "student_performance",
        "Student Performance",
        320,
        "G3",
        "regression",
        649,
        30,
        1,
        (
            "Medu",
            "Fedu",
            "traveltime",
            "studytime",
            "failures",
            "famrel",
            "freetime",
            "goout",
            "Dalc",
            "Walc",
            "health",
        ),
        ("G1", "G2"),
    ),
]

for uci_record in _UCI_CATEGORICAL:
    spec = _uci_categorical(*uci_record)
    DATASETS[spec.name] = spec


DATASETS["south_german_credit"] = DatasetSpec(
    name="south_german_credit",
    title="South German Credit",
    task="classification",
    source_url=(
        "https://archive.ics.uci.edu/static/public/522/south+german+credit.zip"
    ),
    source_filename="uci_522.zip",
    repository_url="https://archive.ics.uci.edu/dataset/522/south+german+credit",
    rows=1_000,
    features=20,
    outputs=2,
    split_kind="stratified",
    citation="South German Credit. UCI Machine Learning Repository, dataset 522.",
    license="CC BY 4.0",
    target="kredit",
    categorical_features=(
        "laufkont",
        "moral",
        "verw",
        "sparkont",
        "beszeit",
        "rate",
        "famges",
        "buerge",
        "wohnzeit",
        "verm",
        "weitkred",
        "wohn",
        "bishkred",
        "beruf",
        "pers",
        "telef",
        "gastarb",
    ),
)

"""Declarative registry of public datasets available to the pipeline."""

from __future__ import annotations

from .models import DatasetSpec

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

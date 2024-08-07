#!/usr/bin/env python
import argparse
import csv
import json
import os
from typing import Set

import numpy as np
import pandas as pd


def main(args: argparse.Namespace) -> None:
    json_files = []

    for root, subdir, files in os.walk(args.json_directory):
        for f in files:
            if f.endswith(".json"):
                path = os.path.join(root, f)
                json_files.append(path)

    if not json_files:
        print("No JSON file is found under " + args.json_directory)
        return

    out = os.path.join(args.output_directory, args.output_name + ".csv")
    with open(out, "w") as fd:
        header: Set[str] = set()
        rows = []
        for f in json_files:
            with open(f, "r") as in_fd:
                b = json.load(in_fd)

            row = {}
            for fk, items in b.items():
                if not isinstance(items, dict):
                    row[fk] = items
                    continue
                for key, val in items.items():
                    row[key] = val

            header = header.union(set(row.keys()))
            rows.append(row)

        writer = csv.DictWriter(fd, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)

    filtered = []
    df = pd.DataFrame(rows, columns=header)

    for i in range(len(df.columns)):
        if not isinstance(df.iloc[0, i], list) and not np.all(
            df.iloc[:, i].values == df.iloc[0, i]
        ):
            filtered.append(df.columns[i])

    filtered = sorted(filtered)
    df = df[filtered]
    df = df.sort_values(filtered, ascending=[False] * len(filtered))
    out = os.path.join(args.output_directory, args.output_name + "-short.csv")
    df.to_csv(out, index=False)
    out = os.path.join(args.output_directory, args.output_name + "-short.xlsx")
    df.to_excel(out, index=False)


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Arguments for converting JSON result to CSV format"
    )
    parser.add_argument("--json-directory", type=str, default=os.path.curdir)
    parser.add_argument("--output-directory", type=str, default=os.path.curdir)
    parser.add_argument("--output-name", type=str, default="out")
    args = parser.parse_args()
    main(args)

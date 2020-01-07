import os
import csv
import json
import argparse


def main(args):
    json_files = []

    for root, subdir, files in os.walk(args.json_directory):
        for f in files:
            if f.endswith('.json'):
                path = os.path.join(root, f)
                json_files.append(path)

    if not json_files:
        print('No JSON file found under ' + args.json_directory)
        return

    out = os.path.join(args.output_directory, args.output_name)
    with open(out, 'w') as fd:
        f = json_files[0]
        fields = ['file']
        with open(f, 'r') as sample_fd:
            b = json.load(sample_fd)
            for key, value in b.items():
                fields = fields + list(value.keys())

        writer = csv.DictWriter(fd, fieldnames=fields)
        writer.writeheader()

        for f in json_files:
            with open(f, 'r') as fd:
                b = json.load(fd)

            row = {}
            for key, value in b['args'].items():
                if value is None:
                    value = 'Null'
                row[key] = value

            for key, value in b[b['args']['algo']].items():
                row[key] = value

            for key, value in b[b['args']['backend']].items():
                row[key] = value

            for key, value in b['packages'].items():
                row[key] = value
            row['file'] = f
            print(row)
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for converting JSON result to CSV format')
    parser.add_argument('--json-directory',
                        type=str,
                        default=os.path.curdir)
    parser.add_argument('--output-directory',
                        type=str,
                        default=os.path.curdir)
    parser.add_argument('--output-name',
                        type=str,
                        default='out.csv')
    args = parser.parse_args()
    main(args)

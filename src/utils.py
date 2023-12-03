import os
import sys


"""

Get relative path with respect to whatever file the calling code is in.

"""


def relative_path(filepath):
    caller__file__ = sys._getframe(1).f_globals["__file__"]
    caller_dirname = os.path.dirname(caller__file__)
    return os.path.join(caller_dirname, filepath)


"""

Converts TSV files to CSV files.

Limitation: will not handle CRLF characters in the TSV, as it reads file line by line.
This is not important for most datasets, and does not negatively affect results using
the data included in this repo.

"""


def convert_tsv_csv(input_filepath, output_filepath):
    with open(output_filepath, "w") as output_file:
        with open(input_filepath) as input_file:
            for line in input_file:
                output_file.write(
                    (
                        ",".join(
                            [
                                '"' + value.replace('"', '""') + '"'
                                for value in line.rstrip().split("\t")
                            ]
                        )
                    )
                    + "\n"
                )

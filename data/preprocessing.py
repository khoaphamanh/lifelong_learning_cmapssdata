import pandas as pd
import os
import re
from tabulate import tabulate
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler


# Preprocessing class
class Preprocessing:
    def __init__(self):

        # seed and name data
        self.seed = 1998
        self.name_data = "CMAPSSData"

        # path
        self.path_data_directory = os.path.dirname(os.path.abspath(__file__))
        self.path_cmapss_directory = os.path.join(
            self.path_data_directory, self.name_data
        )

        # check if data is downloaded
        if not os.path.exists(self.path_cmapss_directory):
            import download_data

        # path files
        self.path_text_files = sorted(
            [
                os.path.join(self.path_cmapss_directory, i)
                for i in os.listdir(self.path_cmapss_directory)
                if i.endswith(".txt") and not i.startswith("readme")
            ]
        )
        self.path_csv_files = [
            os.path.join(
                self.path_cmapss_directory, f"{i.split("/")[-1].split(".")[0]}.csv"
            )
            for i in self.path_text_files
        ]

        # information data
        self.name_subset = sorted(
            list(
                {
                    re.search(r"_(.+)\.", i).group(1)
                    for i in os.listdir(self.path_cmapss_directory)
                    if "FD" in i
                }
            )
        )
        self.type_data = ["train", "test"]

        # save file to csv
        if any([not os.path.exists(i) for i in self.path_csv_files]):
            self.text_to_csv()

    def text_to_csv(
        self,
    ):
        """
        convert from text file to csv
        """
        for path_text_file in self.path_text_files:

            # name text file
            name_text_file = path_text_file.split("/")[-1].split(".")[0]
            # print("name_text_file:", name_text_file)

            # convert to df
            df = pd.read_csv(path_text_file, sep=r"\s+", header=None)

            # save as csv
            path_csv_file = os.path.join(
                self.path_cmapss_directory, f"{name_text_file}.csv"
            )

            df.to_csv(path_csv_file, index=False)


if __name__ == "__main__":

    preprocessing = Preprocessing()

    text_to_csv = preprocessing.text_to_csv()
    # table_analysis = preprocessing.table_analysis()
    plot_ts = preprocessing.plot_ts(
        name_subset="FD001", type_data="train", engine=1, feature=None, normalize=True
    )

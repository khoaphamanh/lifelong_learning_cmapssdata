import pandas as pd
import os
import re
from tabulate import tabulate
import numpy as np


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
        self.name_engine = sorted(
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
        if not any([os.path.exists(i) for i in self.path_csv_files]):
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

    def table_analysis(self):
        """
        create table for analysis
        """
        dict_analysis_file = {"name": [], "number_of_unit": [], "shape": []}
        dict_analysis_unit = {}

        for path_csv_file in self.path_csv_files:

            # df
            df = pd.read_csv(path_csv_file)

            # name of csv file
            name_file = path_csv_file.split("/")[-1].split(".")[0]
            dict_analysis_file["name"].append(name_file)

            # shape
            dict_analysis_file["shape"].append(df.shape)

            # number of unit and number instance of each unit
            if name_file.startswith("RUL"):
                dict_analysis_file["number_of_unit"].append(len(df))
                dict_analysis_unit[name_file] = df.iloc[:, 0].tolist()

            else:
                dict_analysis_file["number_of_unit"].append(df.iloc[:, 0].nunique())
                unit, count = np.unique(df.iloc[:, 0], return_counts=True)
                dict_analysis_unit[name_file] = count.tolist()

        # append to have all same len values for dict_analysis_unit
        max_len_values_dict_unit = max([len(i) for i in dict_analysis_unit.values()])
        dict_analysis_unit["1unit"] = list(range(max_len_values_dict_unit))
        for k, v in dict_analysis_unit.items():
            dict_analysis_unit[k] = v + [None] * (max_len_values_dict_unit - len(v))
        dict_analysis_unit = {
            k: dict_analysis_unit[k] for k in sorted(dict_analysis_unit.keys())
        }

        # create tabular
        tabular_file = tabulate(tabular_data=dict_analysis_file, headers="keys")
        tabular_unit = tabulate(tabular_data=dict_analysis_unit, headers="keys")

        return tabular_file, tabular_unit


if __name__ == "__main__":

    preprocessing = Preprocessing()

    # text_to_csv = preprocessing.text_to_csv()
    table_analysis = preprocessing.table_analysis()

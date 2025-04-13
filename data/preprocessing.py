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

        # preprocessed data
        self.path_preprocessed_data_directory = os.path.join(
            self.path_data_directory, "preprocessed"
        )

        self.name_subsets = sorted(
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

    def column_constant(self):
        """
        find the column with constant values
        """
        # column constant dict
        column_constant_dict = {
            "FD001": ["4", "5", "9", "10", "14", "20", "22", "23"],
            "FD002": ["4", "17", "23"],
            "FD003": ["4", "5", "9", "10", "14", "20", "22", "23"],
            "FD004": ["4", "17", "23"],
        }

        return column_constant_dict

    def load_raw_df_data(self, normalize=False):
        """
        load raw data and save in a dict (and normalize)
        """
        # data dict
        train_data = {}
        test_data = {}
        test_rul = {}

        # path of train, test and rul
        path_train_subset_csv = [i for i in self.path_csv_files if "train" in i]
        path_test_subset_csv = [i for i in self.path_csv_files if "test" in i]
        path_test_rul_csv = [i for i in self.path_csv_files if "RUL" in i]

        # loop through all path
        for train_csv, test_csv, rul_csv, name_subset in zip(
            path_train_subset_csv,
            path_test_subset_csv,
            path_test_rul_csv,
            self.name_subsets,
        ):
            # read csv
            df_raw_train = pd.read_csv(train_csv)
            df_raw_test = pd.read_csv(test_csv)
            df_rul = pd.read_csv(rul_csv)

            # eliminate the constant columns
            columns_elim = self.column_constant()[name_subset]
            df_raw_train = df_raw_train.drop(columns=columns_elim)
            df_raw_test = df_raw_test.drop(columns=columns_elim)

            # engine
            engines = df_raw_train.iloc[:, 0].unique()
            df_train_scaled = []
            df_test_scaled = []

            for eng in engines:

                # find the indices for each engine
                indices_eng_train = np.where(df_raw_train.iloc[:, 0] == eng)[0]
                # print("indices_eng_train:", indices_eng_train)
                # print(eng)
                indices_eng_test = np.where(df_raw_test.iloc[:, 0] == eng)[0]
                # print(eng)
                # print("indices_eng_test shape:", indices_eng_test.shape)
                # print("indices_eng_test:", indices_eng_test)

                # indexing each engine
                df_eng_train = df_raw_train.iloc[indices_eng_train, :]
                df_eng_test = df_raw_test.iloc[indices_eng_test, :]

                # scale each engine
                if len(indices_eng_test) > 0:
                    if normalize:
                        scaler = StandardScaler()
                        colums_to_scale = df_raw_train.columns[2:]

                        # copy the data
                        df_eng_train_scaled = df_eng_train.copy()
                        df_eng_test_scaled = df_eng_test.copy()

                        df_eng_train_scaled[colums_to_scale] = scaler.fit_transform(
                            df_eng_train_scaled[colums_to_scale]
                        )
                        df_eng_test_scaled[colums_to_scale] = scaler.transform(
                            df_eng_test_scaled[colums_to_scale]
                        )

                    else:
                        df_eng_train_scaled = df_eng_train
                        df_eng_test_scaled = df_eng_test

                    # append to a big scaled list
                    df_train_scaled.append(df_eng_train_scaled)
                    df_test_scaled.append(df_eng_test_scaled)

            # concat the df
            df_train_scaled = pd.concat(df_train_scaled, ignore_index=True)
            df_test_scaled = pd.concat(df_test_scaled, ignore_index=True)

            # save to dict
            train_data[name_subset] = df_train_scaled
            test_data[name_subset] = df_test_scaled
            test_rul[name_subset] = df_rul

        return train_data, test_data, test_rul

    def windowing(self, window_size=30, hop_size=1, normalize=False):
        """
        windowing each engine
        """
        # windows dict
        X_train = {}
        y_train = {}
        X_test = {}
        y_test = {}
        n_windows_pro_train_engines = {}

        # load raw data df
        train_data, test_data, test_rul = self.load_raw_df_data(normalize=normalize)

        # windowing for each engine in train data
        data_list = [train_data, test_data]

        # loop for each data
        for idx_d, data_dict in enumerate(data_list):
            # loop for each subset
            for name_subset, df_data in data_dict.items():
                engines = df_data.iloc[:, 0].unique()

                # add keys as name subset to windows dict
                if idx_d == 0:
                    X_train[name_subset] = []
                    y_train[name_subset] = []
                    n_windows_pro_train_engines[name_subset] = []

                else:
                    X_test[name_subset] = []
                    y_test[name_subset] = []

                # loop for each engine
                for eng in engines:
                    indices_engine = np.where(df_data.iloc[:, 0] == eng)[0]
                    # print("indices_engine:", indices_engine)
                    df_engine = df_data.iloc[indices_engine, 2:]
                    # print(eng)
                    # print(name_subset)
                    # print("df_engine shape:", df_engine.shape)

                    # number of windows
                    n_rows = len(df_engine)

                    # padding if n_rows smaller than window size
                    if n_rows < window_size:
                        n_cols = df_engine.shape[1]
                        df_pad = pd.DataFrame(
                            [[0] * (n_cols)] * (window_size - n_rows),
                            columns=df_engine.columns,
                        )
                        df_engine = pd.concat([df_pad, df_engine], ignore_index=True)

                    # windowing and find the correspond rul
                    n_rows = len(df_engine)
                    n_windows = (n_rows - window_size) // hop_size + 1
                    windows = np.array(
                        [
                            df_engine.iloc[i * hop_size : i * hop_size + window_size, :]
                            for i in range(n_windows)
                        ]
                    )

                    ruls = np.array(
                        [
                            n_rows - (i * hop_size + window_size)
                            for i in range(n_windows)
                        ]
                    )

                    # ruls for test data
                    ruls = (
                        ruls + test_rul[name_subset].iloc[eng - 1, 0]
                        if idx_d == 1
                        else ruls
                    )

                    # append to dict
                    if idx_d == 0:
                        X_train[name_subset].append(windows)
                        y_train[name_subset].append(ruls)
                        n_windows_pro_train_engines[name_subset].append(n_windows)

                    else:
                        X_test[name_subset].append(windows)
                        y_test[name_subset].append(ruls)

                # concatnate for all engines in a same subset
                if idx_d == 0:
                    X_train[name_subset] = np.vstack(X_train[name_subset])
                    y_train[name_subset] = np.concatenate(y_train[name_subset])

                else:
                    X_test[name_subset] = np.vstack(X_test[name_subset])
                    y_test[name_subset] = np.concatenate(y_test[name_subset])

        return X_train, X_test, y_train, y_test, n_windows_pro_train_engines

    def get_n_check_points_pro_engine(self, n_checkpoint, ratios_dict):
        """
        find the number of checkpoints based on number of windows in each engine
        """
        # number of checkpoints pro engine dict
        n_checkpoints_pro_engines_dict = {}

        # number of checkpoints per engine
        for name_subset, ratios in ratios_dict.items():

            # total ratios
            total_ratio = sum(ratios)

            # number of checkpoints per engine each subset
            n_checkpoints_pro_engines = np.floor(
                [(r / total_ratio * n_checkpoint) for r in ratios]
            )

            # save to dict
            n_checkpoints_pro_engines_dict[name_subset] = n_checkpoints_pro_engines

        return n_checkpoints_pro_engines_dict


if __name__ == "__main__":

    from timeit import default_timer

    start = default_timer()
    preprocessing = Preprocessing()

    # text_to_csv = preprocessing.text_to_csv()
    # table_analysis = preprocessing.table_analysis()
    # plot_ts = preprocessing.plot_ts(
    #     name_subset="FD001", type_data="train", engine=1, feature=None, normalize=True
    # )

    # path_csv_file = preprocessing.path_csv_files
    # print("path_csv_file len:", len(path_csv_file))
    # print("print:", print)
    # print("path_csv_file:", path_csv_file)

    # column_constant = preprocessing.column_constant()
    # print("column_constant:", column_constant)

    # print("scaler:", scaler)

    # load_raw_data = preprocessing.load_raw_df_data(normalize=False)

    X_train, X_test, y_train, y_test, n_windows_pro_train_engines = (
        preprocessing.windowing(normalize=True)
    )

    # for i, j in X_train.items():
    #     print(i, j.shape)

    # for i, j in n_windows_pro_train_engines.items():
    #     print(i, sum(j))
    # # for i in X_test["FD002"]:
    #     if np.any(i == 0):
    #         print(i)

    # a = X_test["FD002"][-1]
    # print("a:", a)
    # b = y_test["FD002"][-1]
    # print("b:", b)

    # b = y_test["FD004"]
    # print("b:", b)
    n_checkpoints_pro_engines_dict = preprocessing.get_n_check_points_pro_engine(
        n_checkpoint=100, ratios_dict=n_windows_pro_train_engines
    )
    for k, v in n_checkpoints_pro_engines_dict.items():
        print(k)
        print(v)
        print(sum(v))

    end = default_timer()
    print(end - start)

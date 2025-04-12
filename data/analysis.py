from preprocessing import Preprocessing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os


# Analysis class
class Analysis(Preprocessing):
    def __init__(self):
        super().__init__()

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
        dict_analysis_unit["1unit"] = list(range(1, max_len_values_dict_unit + 1))
        for k, v in dict_analysis_unit.items():
            dict_analysis_unit[k] = v + [None] * (max_len_values_dict_unit - len(v))
        dict_analysis_unit = {
            k: dict_analysis_unit[k] for k in sorted(dict_analysis_unit.keys())
        }

        # create tabular
        tabular_file = tabulate(tabular_data=dict_analysis_file, headers="keys")
        tabular_unit = tabulate(tabular_data=dict_analysis_unit, headers="keys")

        return tabular_file, tabular_unit

    def load_visualize_df(
        self, name_subset, type_data, engine, feature=None, normalize=False
    ):
        """
        load the ts given name_subset, type_data, engine, feature, normalize
        """
        # load data csv
        path_csv = os.path.join(
            self.path_cmapss_directory, f"{type_data}_{name_subset}.csv"
        )
        df = pd.read_csv(path_csv)

        # check if plot single or multiple engine
        indices_engine = np.where(df.iloc[:, 0] == engine)[0]
        df_engine = df.iloc[indices_engine, 2:]
        df_engine = (
            df_engine.iloc[:, [feature - 2]]
            if feature not in [None, "all"]
            else df_engine
        )

        # normalize
        scaler = StandardScaler()
        df_engine = (
            pd.DataFrame(scaler.fit_transform(df_engine), columns=df_engine.columns)
            if normalize
            else df_engine
        )

        return df_engine

    def load_visualize_ts_test(self, name_subset):
        """
        load test file RUL_name_sub_set.csv
        """
        # load RUL_name_sub_set.csv file
        path_csv_test = os.path.join(
            self.path_cmapss_directory, f"RUL_{name_subset}.csv"
        )
        df_test = pd.read_csv(path_csv_test)

        return df_test

    def plot_visualize_feature_histogram(
        self, name_subset, type_data, engine, normalize=False
    ):
        """
        plot the histogram of each feature
        """
        # load df_engine
        df_engine = self.load_visualize_df(
            name_subset=name_subset,
            type_data=type_data,
            engine=engine,
            feature=None,
            normalize=normalize,
        )

        # plot the histogram
        rows, cols = 6, 4
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=df_engine.columns)

        # Add a histogram for each feature
        for i, col in enumerate(df_engine.columns):
            r = i // cols + 1  # row index
            c = i % cols + 1  # col index
            fig.add_trace(
                go.Histogram(x=df_engine[col], name=col, showlegend=False), row=r, col=c
            )

        # RUL
        RUL = 0
        if type_data == "test":
            df_test = self.load_visualize_ts_test(name_subset=name_subset)
            RUL = df_test.iloc[engine, 0]

        title = f"Subset {name_subset}, engine {engine}, feature all, normalize {normalize}, n_cycle {len(df_engine)}, RUL {RUL}"

        # add layout
        fig.update_layout(
            height=1500,
            width=1200,
            title_text=title,
            template="plotly_white",
        )

        # show the fig
        fig.show()

        return fig

    def plot_visulize_ts(
        self, name_subset, type_data, engine, feature=None, normalize=False
    ):
        """
        plot ts given name_subset, type_data, engine, feature, normalize
        """
        # load df_engine
        df_engine = self.load_visualize_df(
            name_subset=name_subset,
            type_data=type_data,
            engine=engine,
            feature=feature,
            normalize=normalize,
        )

        # plot the image
        fig = go.Figure()
        cycle = list(range(len(df_engine)))

        for f, c in enumerate(df_engine.columns):
            value = df_engine[c]
            fig.add_trace(
                go.Scatter(
                    x=cycle,
                    y=value,
                    mode="lines",
                    name=f"Feature {c}",
                    showlegend=True,
                )
            )

        # RUL
        feature = "all" if feature == None else feature

        # add layout
        RUL = 0
        if type_data == "test":
            df_test = self.load_visualize_ts_test(name_subset=name_subset)
            RUL = df_test.iloc[engine, 0]

        # add layout
        title = f"Subset {name_subset}, engine {engine}, feature {feature}, normalize {normalize}, n_cycle {len(cycle)}, RUL {RUL}"

        fig.update_layout(
            title=title,
            xaxis_title="Cycle",
            yaxis_title="Value",
            legend_title="Feature",
            template="plotly_white",
        )

        # check constant column
        column_constant = df_engine.columns[df_engine.nunique() == 1]
        print("Constant columns:", list(column_constant))

        # show the image
        fig.show()

        return fig


if __name__ == "__main__":

    analyxix = Analysis()

    # text_to_csv = analyxix.text_to_csv()

    plot_ts = analyxix.plot_visulize_ts(
        name_subset="FD001", type_data="train", engine=1, feature=None, normalize=True
    )

    plot_histogram = analyxix.plot_visualize_feature_histogram(
        name_subset="FD001", type_data="train", engine=1, normalize=True
    )

import numpy as np
import pandas as pd
import seaborn





def to_df(dataset):
    return pd.DataFrame(
        data=np.c_[dataset["data"], dataset["target"]],
        columns=dataset["feature_names"] + ["target"],
    )


def load_dataset(dataset_name: str):
    if dataset_name == "titanic":
        dataset = seaborn.load_dataset("titanic")
        # Replace 'survived' column with 'target' for consistency
        dataset["target"] = dataset["survived"]
        dataset = dataset.drop(columns=["survived"])
        dataset["target"] = dataset["target"].astype(int)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

    X = dataset.drop(columns=["target"])
    y = dataset["target"]
    return X, y


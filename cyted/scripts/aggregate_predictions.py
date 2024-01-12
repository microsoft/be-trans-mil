#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""
This script aggregates cross validation predictions from the AzureML runs of the cyted model and saves them to a
csv file along with the corresponding metadata. It takes a parent_run_id as well as any additional run ids to include
in the aggregation in case some child runs failed. The parent run id is the run id of the Hyperdrive run that was used
to train the model. The script will download the predictions from the AzureML run outputs and aggregate them into a
single csv file for each of test and validation stages.
"""
from enum import Enum
import pandas as pd
from argparse import ArgumentParser
from azureml._restclient.constants import RunStatus
from cyted.cyted_schema import CytedSchema
from cyted.data_paths import CYTED_CSV_FILENAME, CYTED_RAW_DATASET_DIR
from cyted.fixed_paths import repository_root_directory
from cyted.utils.azure_utils import EXTRA_VAL, TEST, DictPathsType, download_outputs_csv_from_runs, get_target_runs
from health_cpath.utils.naming import ResultsKey
from pathlib import Path
from typing import List, Optional


class ClassPrediction(str, Enum):
    """Enum for the class prediction."""

    TRUE_POS = "true_pos"
    TRUE_NEG = "true_neg"
    FALSE_POS = "false_pos"
    FALSE_NEG = "false_neg"


class CVPredictionsAggregator:
    """Class to aggregate cross validation predictions from multiple runs."""

    def __init__(
        self,
        output_dir: Path,
        outputs_csv_paths: DictPathsType,
        stages: List[str] = [TEST, EXTRA_VAL],
        dataset_csv: Path = Path(CYTED_RAW_DATASET_DIR) / CYTED_CSV_FILENAME,
    ) -> None:
        """
        :param output_dir: The directory to save the aggregated predictions to.
        :param outputs_csv_paths: The paths to the csv files containing the predictions from each run.
        :param stages: The stages to aggregate predictions for. Defaults to [TEST, EXTRA_VAL]
        :param dataset_csv: The path to the csv file containing the metadata for the dataset.
        """
        self.output_dir = output_dir
        self.outputs_csv_paths = outputs_csv_paths
        self.stages = stages
        self.dataset_df = pd.read_csv(dataset_csv, sep="\t")

    def select_class_predictions_from_df(self, df: pd.DataFrame, class_pred: str) -> pd.DataFrame:
        """Select the class predictions from the given dataframe.

        :param df: The dataframe to select the class predictions from.
        :param class_pred: The class prediction to select.
        :raises ValueError: If the class prediction is invalid.
        :return The dataframe containing only the selected class predictions.
        """
        if class_pred == ClassPrediction.TRUE_POS:
            df = df[df[ResultsKey.TRUE_LABEL] == 1]
            df = df[df[ResultsKey.PRED_LABEL] == 1]
        elif class_pred == ClassPrediction.TRUE_NEG:
            df = df[df[ResultsKey.TRUE_LABEL] == 0]
            df = df[df[ResultsKey.PRED_LABEL] == 0]
        elif class_pred == ClassPrediction.FALSE_POS:
            df = df[df[ResultsKey.TRUE_LABEL] == 0]
            df = df[df[ResultsKey.PRED_LABEL] == 1]
        elif class_pred == ClassPrediction.FALSE_NEG:
            df = df[df[ResultsKey.TRUE_LABEL] == 1]
            df = df[df[ResultsKey.PRED_LABEL] == 0]
        else:
            raise ValueError(f"Invalid class prediction {class_pred}")
        return df

    def select_predictions_from_csv(
        self, download_dir: Path, outputs_csv_path: Path, class_prediction: ClassPrediction
    ) -> None:
        """Select the predictions from the given outputs csv file and save them to a new csv file.

        :param download_dir: The directory to save the new csv file to.
        :param outputs_csv_path: The path to the outputs csv file.
        :param class_prediction: The class prediction to select from the outputs csv file.
        """
        df = pd.read_csv(
            outputs_csv_path,
            usecols=[
                ResultsKey.SLIDE_ID,
                ResultsKey.TRUE_LABEL,
                ResultsKey.PRED_LABEL,
                f"{ResultsKey.CLASS_PROBS}0",
                f"{ResultsKey.CLASS_PROBS}1",
            ],
        )
        df.drop_duplicates(subset=ResultsKey.SLIDE_ID, inplace=True)
        df = self.select_class_predictions_from_df(df, class_prediction)
        df = self.dataset_df.merge(df, right_on=ResultsKey.SLIDE_ID, left_on=CytedSchema.CytedID)
        df.to_csv(download_dir / f"{outputs_csv_path.stem}.csv", index=False)

    def select_and_save_predictions_csvs(
        self,
        output_dir: Path,
        class_prediction: ClassPrediction,
    ) -> DictPathsType:
        """Select and save the predictions from the given outputs csv files to new csv files.

        :param output_dir: The directory to save the new csv files to.
        :param class_prediction: The class prediction to select from the outputs csv files.
        :return: A dictionary mapping the stages to the paths of the new csv files containing the predictions and
            corresponding metadata.
        """
        predictions_csvs: DictPathsType = {stage: [] for stage in self.stages}
        for stage in self.stages:
            for outputs_csv_path in self.outputs_csv_paths[stage]:
                self.select_predictions_from_csv(output_dir, outputs_csv_path, class_prediction)
                predictions_csvs[stage].append(output_dir / f"{outputs_csv_path.stem}.csv")
        return predictions_csvs

    def merge_predictions_csvs(
        self,
        output_dir: Path,
        predictions_csvs: DictPathsType,
        class_prediction: ClassPrediction,
    ) -> None:
        """Merge the predictions from the given csv files into a single csv file for each stage.

        :param output_dir: The directory to save the new csv files to.
        :param predictions_csvs: The paths to the csv files containing the predictions.
        :param class_prediction: The class prediction to select from the outputs csv files.
        """
        for stage in self.stages:
            dfs = [pd.read_csv(path) for path in predictions_csvs[stage]]
            df = pd.concat(dfs)
            prob_class1 = df.groupby(CytedSchema.CytedID).prob_class1.mean()
            df.drop_duplicates(subset=CytedSchema.CytedID, inplace=True)
            df.set_index(CytedSchema.CytedID, inplace=True)
            df.prob_class1 = prob_class1
            df.prob_class0 = 1 - df.prob_class1
            df.pred_label = (df.prob_class1 >= df.prob_class0).astype(int)
            print(f"Number of {class_prediction} for {stage}: {len(df)} saved to {f'{stage}.csv'}")
            df.to_csv(output_dir / f"{stage}.csv")

    def aggregate_predictions(self) -> None:
        """Aggregate the predictions from the given outputs csv files and save them to new csv files."""
        for class_prediction in ClassPrediction:
            print(f"Aggregating {class_prediction.value} predictions")
            output_dir = self.output_dir / class_prediction.value
            output_dir.mkdir(exist_ok=True)
            predictions_csvs = self.select_and_save_predictions_csvs(output_dir, class_prediction)
            self.merge_predictions_csvs(output_dir, predictions_csvs, class_prediction)


def main(
    download_dir: Path,
    dataset_csv: Path,
    stages: List[str],
    parent_run_id: str,
    child_runs_status: Optional[str] = None,
    additional_run_ids: Optional[List[str]] = None,
    workspace_config: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    """Aggregate the predictions from the AzureML runs of the cyted model and save them to a csv file
    along with the corresponding metadata for each of specified stages.

    :param download_dir: The directory to save the new csv files to.
    :param dataset_csv: The path to the dataset csv file.
    :param stages: The stages to aggregate the predictions for e.g. test and validation.
    :param parent_run_id: The run id of the Hyperdrive run that was used to train the model.
    :param child_runs_status: The status of the child runs to aggregate the predictions from.
    :param additional_run_ids: Any additional run ids to aggregate the predictions from.
    :param workspace_config: The path to the workspace config file.
    :param overwrite: Whether to overwrite the existing pre downloaded output csv files.
    """

    runs = get_target_runs(
        parent_run_id=parent_run_id,
        child_runs_status=child_runs_status,
        additional_run_ids=additional_run_ids,
        workspace_config=workspace_config,
    )

    output_dir = download_dir / "outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    outputs_csv_paths = download_outputs_csv_from_runs(output_dir, runs=runs, overwrite=overwrite, stages=stages)

    aggregator = CVPredictionsAggregator(
        output_dir=download_dir, outputs_csv_paths=outputs_csv_paths, dataset_csv=dataset_csv, stages=stages
    )
    aggregator.aggregate_predictions()


if __name__ == "__main__":
    """
    Usage example from CLI:
    python cyted_aggregate_predictions.py \
    --run_id <insert AML run ID here> \
    --additional_run_ids <insert additional AML run IDs here> \
    --workspace_config config.json
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--run_id", help="The parent Hyperdrive run ID."
    )
    parser.add_argument(
        "--child_runs_status",
        default=RunStatus.COMPLETED,
        help="The status of the child runs to download. Default is 'Completed'."
        "Available options: 'Completed', 'Failed', 'Running'.",
    )
    parser.add_argument("--additional_run_ids", default=None, help="List of run ids, separated by comma.")
    parser.add_argument(
        "--workspace_config",
        default="config.json",
        help="Path to Azure ML workspace config.json file. If omitted, will try default workspace.",
    )
    parser.add_argument("--download_dir", help="Directory where to download Azure ML data.")
    parser.add_argument(
        "--dataset_csv", default=Path(CYTED_RAW_DATASET_DIR) / CYTED_CSV_FILENAME, help="Dataset CSV file"
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Forces (re)download of metrics and output files, even if they already exist locally.",
    )
    args = parser.parse_args()

    if args.download_dir is None:
        download_dir = repository_root_directory("outputs") / "cv_predictions" / args.run_id
    else:
        download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    workspace_config = Path(args.workspace_config).resolve() if args.workspace_config else None
    additional_run_ids = args.additional_run_ids.split(",") if args.additional_run_ids else None

    print(f"Download dir: {download_dir}")
    print(f"Workspace config: {workspace_config}")
    main(
        download_dir=download_dir,
        dataset_csv=args.dataset_csv,
        stages=[TEST, EXTRA_VAL],
        parent_run_id=args.run_id,
        child_runs_status=args.child_runs_status,
        additional_run_ids=additional_run_ids,
        workspace_config=workspace_config,
        overwrite=args.overwrite,
    )

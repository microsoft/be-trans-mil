
# Workflow for Training and Testing TFF3 Models

This documents contains detailed instructions to reproduce TFF3 prediction models from TFF3 slides. These are the expected results.

| Model Description |                                      |                   |                     | Metrics            |                 |                 |                 |
| ----------------- | ------------------------------------ | ----------------- | ------------------- | ------------------ | --------------- | --------------- | --------------- |
| Encoder           | Pooling                              | Bag size training | Bag size validation | Accuracy           | Auroc           | Specificity     | Recall          |
| Resnet 18         | 4 Transformer layers + Attention MIL |       2300        |       4600          | V: 0.8684 ± 0.0208 | 0.9144 ± 0.0179 | 0.9131 ± 0.0476 | 0.7959 ± 0.0426 |
|                   |                                      |                   |                     | T: 0.8376 ± 0.0184 |                 |                 |                 |
|                   |                                      |                   |                     | E: 0.8646          |                 |                 |                 |
| Resnet 50         | 4 Transformer layers + Attention MIL |       1700        |       3400          |                    |                 |                 |                 |
| Swin T            | 4 Transformer layers + Attention MIL |       1000        |       2000          |                    |                 |                 |                 |

Bag size stands for the number of tiles sampled from the slide on the fly.

V: stands for cross validation results, T: average test performance and E: ensembled test results across all folds using a majority vote.

TODO: Update table results with final values

# TFF3 Data preprocessing

First, make sure you have followed Azure instructions in `docs/workflow.md`
to be able to run preprocessing in the cloud.

Follow these steps to create the preprocessed whole slide dataset.

## Exclude control tissue using AML Labelling

Control tissue is excluded from TFF3 slides using AzureML automatic labelling for object detection.
[(Set up image labelling)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-image-labeling-projects)
Tissue sections are segmented by drawing bounding boxes around each section.

Bounding boxes for Delta cohort (`tff3_control_tissue_exclusion_via_aml_labelling.json`) are available upon request from Cyted.

## Crop the TFF3 Slides

The next step is to reduce the size of the TFF3 slides, by taking only the two tissue sections that are of interest.
This uses the bounding boxes that were extracted in the previous step.

- In addition, the script re-writes the NDPI files as TIFF files with only one specific magnification level. This greatly reduces
  the file size, and consequently the time it takes to load them for training.
- The script also standardizes the background of the slides, by setting all pixels that are not in the tissue sections to white.
  The decision about foreground/background is based on the masks that are created by a segmentation algorithm. The algorithm considers each tissue section at 1.25x magnification, extracts hematoxylin channel using background as median of the image, and thresholds the image using a quantile (0.8 by default) followed by morphological closing and opening operations (structuring element 10 px by default). The created masks are used to hardcode background of the TFF3 slides to white.

Run the following commandline to create preprocessed TFF3 dataset at 10x in azure.

```shell
python cyted/preproc/crop_and_convert_ndpi_to_tiff.py \
  --image_column=TFF3 \
  --dataset=cyted-raw-20221102 \
  --dataset_csv=reviewed_dataset_2023-02-10.tsv \
  --output_mask_dataset=masks_ \
  --excluded_slides_csv=cyted/preproc/files/tff3_exclusion_list.csv \
  --converted_dataset_csv=dataset.tsv \
  --bounding_box_path=cyted/preproc/files/tff3_control_tissue_exclusion_via_aml_labelling.json  \
  --target_magnifications=10.0 \
  --num_workers=20 \
  --cluster=<cluster_name> \
  --datastore=<datastore_name> \
  --automatic_output_name=True \
  --display_name=preprocess_tff3_slides \
  --docker_shm_size=<vn_ram_size>
```

This scripts runs directly in AzureML so the preprocessed data will be stored in blob storage automatically.

For a full documentation of the parameters, run `python cyted/preproc/crop_and_convert_ndpi_to_tiff.py --help`

We recommand setting `--output_dataset` for the delta cohort to match the dataset names set in `cyted/data_paths.py`

# Running Training

To run training, you need to have a GPU-enabled AzureML compute cluster. For local development work, a workstation or VM with a GPU
can be utilized too.

To create a compute cluster, see the [hi-ml docs](https://hi-ml.readthedocs.io/en/latest/azure_setup.html) for instructions.

Once a compute cluster is available, you can run training as follows:

```shell
python runner.py --model cyted.TFF3CytedMIL --cluster=mygpucluster
```

(Replace `mygpucluster` with the name of your compute cluster.)

All model and training parameters can be changed on the command line. See `python runner.py --help` for a list of all parameters.
Here are a few particularly helpful ones:

- `--max_epochs=1`: Only run for one epoch. This is helpful to speed up training runs for debugging purposes.
- `--crossval_count=5`: Start 5-fold cross validation. This will queue 5 training runs, each with a different fold as validation set.
- `--pl_limit_train_batches=2`: Limit the number of training batches to 2. Again, this is helpful for debugging.
- `--max_bag_size=3`: Change the number of tiles in a bag during training.
- `--max_bag_size_inf=3`: Change the number of tiles in a bug during whole slide inference
- `--batch_size=2`: Change the number of tiles in a bag during training.
- `--batch_size_inf=2`: Change the number of tiles in a bag during whole slide inference.

# Workflow for Slide Quality Control

## Group Slides by Stain Type

We can group slides by stain type by considering all files that are contained in the "H&E" (or TFF3) column of the
metadata file. This is done by creating a folder for each stain type, and creating symbolic links to the slides in their
original location.

- For grouping slides by type, you need to first mount the datasets container as described in [azure_setup.md](azure_setup.md).
  This will make the data available at the path `/cyted`
- `setup/group_slides.sh` creates 3 folders with slides grouped by stain type (one folder for H&E, one for TFF3, one for
  P53). The script reads a datafile with one row per slide.
- The script has its parameters set to, by default, process a variant of the `delta` dataset,
  `delta_slides_msr_2022-11-11.tsv`. In this metadata file, we have already corrected some labelling mistakes (TFF3
  slides contained in the H&E column, or vice verse). If you want to process a different file, you need to manually
  modify the `DATA` and `INPUT_FILE` variables in the script to point to the correct folders and dataset files.
- Run the script `setup/group_slides.sh`. It will create a folder for each stain type, with symbolic links to the slides
  in their original location.
- The folder with only the H&E slids will be in `~/cyted-temp/he_slides_only`, the folder with only the TFF3 slides will be in
  `~/cyted-temp/tff3_slides_only`

## Prepare a Montage of Slides per Stain Type

A montage is a single image that contains thumbnails of all the slides in a dataset. It is useful for visual inspection
of the dataset.

- To create a montage, we first extract all slides for a given stain (H&E, TFF3, P53) into folder, by creating
  symbolic links. This is done by running `setup/group_slides.sh`, see instructions above.
- Consider renaming the folder with the grouped slides. By default, it is `~/cyted-temp/he_slides_only`, meaning that the
  folder in Azure blob storage will also be called `he_slides_only`. For the `best2` dataset, you can name the folder
  `best2_he_slides_only`.
- Upload the folders with the grouped slides to the Azure Storage Account. See the instructions in
  [azure_setup.md](azure_setup.md).
- Run montage creation in the cloud, following the instructions in [hi-ml docs](https://hi-ml.readthedocs.io/en/latest/montage_creation.html)

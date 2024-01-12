# Pre-processing scripts for Cyted data

This folder contains scripts that were used to pre-process the Cyted data. Most scripts are used as one-off scripts that read an existing datafile,
and write a modified one back to the dataset.

## Dataset mounting setup

The Cyted dataset container must be mounted to the `/cyted` folder. For that, you need to follow the instructions in `docs/workflow.md`.

This should be run on a VM that is in the same location as the storage account,
to reduce latency.

The modified data files will be written back to the storage container.

## Data Conversion from NDPI to TIFF

The script `convert_cyted_ndpi_to_tiff.py` allows to convert cyted whole slides, originally in ndpi format, to tiff files. Use the following commandline to launch the data conversion:

```bash
cd cyted/preproc
python convert_cyted_ndpi_to_tiff.py --root_dataset=<your_root> --target_magnification=<your_target_mag> --num_workers=<n> --dest_dir=<your_dest_dir> --label_col=<label> --image_col=<HE>
```

The converted files are pyramidal tiff files containing 2 levels:

- level 0 corresponds to the target magnification
- level 1 stores the lowest magnification available in the original ndpi file. This is useful for fast foreground selection when loading the slides.

Additionaly, the scripts generates a new tsv file with updated file names (.tiff extension and replacement of `&` by `_`) and the exact same metadata. The new metadata will have the same name as in the original ndpi root directory and will be saved in the dest dictionary.

To further validate that the new data is not corrupted[^1], we recommend inspecting the new files size:

```bash
du -sh dest_dir/*
```

The new files should in the order of 200MB to 1200MB. Detect any outliers using the following command that returns files <100MB

```bash
find . -type f -size -100M
```

[^1]: Causes of corruption includes but not limited to: sudden preemption of the conversion script, ...

# Setting up Azure and Data Handling

This document describes

- How to set up Azure
- How to transfer data to/from Azure
- How to make data in Azure available on your workstation or Azure VM

## Setting up Azure

You will need:

- An Azure subscription
- Quota to use GPUs
- An Azure Storage Account that holds the data
- An AzureML Workspace
- Create compute clusters

Detailed instructions and a script-driven setup are available in the [hi-ml documentation](https://hi-ml.readthedocs.io/en/latest/azure_setup.html).

- As the last step of the setup, download the `config.json` file from the AzureML workspace and place it in the `cyted-msr` repository root.
- In your storage account, create a container called `datasets`, see the
[docs](https://learn.microsoft.com/en-us/azure/storage/blobs/blob-containers-portal)
- You now need to modify the Access Control section of your storage account: Click "+ Add", "Add role assignment". In
  the list of available roles, highlight the "Storage Blob Data Owner", press Next. Then select all
  the people to add with those permissions.

To create a compute cluster, see the [hi-ml docs](https://hi-ml.readthedocs.io/en/latest/azure_setup.html) for instructions.

## Uploading data to Azure

Uploading a folder of data to Azure Storage can be done as follows:

1. Download and extract `azcopy` via `wget https://aka.ms/downloadazcopy-v10-linux`
1. Extract the tarball via `tar xvf downloadazcopy-v10-linux`. This will create a folder with a single executable `azcopy`.
    `cd` into that folder, or copy `azcopy` to a folder that is in your `PATH`.
1. Log into Azure via `azcopy login`
1. Optional: If the `copy` operations says that "failed to get keyring during save", then install
    the `keyctl` tool (via `sudo apt-get install keyutils`) and start a new session
    via `keyctl session anysessionname`.

Then run the `copy` command like this, replacing the source folder with the folder you want to upload, and `<blob_account_name>` with the name of your storage account:

```azcopy copy /data/a_folder_to_copy 'https://<blob_account_name>.blob.core.windows.net/datasets' --recursive```

This will create a folder `a_folder_to_copy` (or whatever you called it) in the `datasets` container in your storage
account.

## Copying from AWS to Azure

You can also copy directly from AWS to Azure. For that to work, you need to log into Azure as described in the previous
section. You also need to set up an access key for your S3 bucket in AWS. Contact your local AWS expert to set
those up. Put the access key and its associated secret into these two environment variables:

```bash
export AWS_ACCESS_KEY_ID=<access-key>
export AWS_SECRET_ACCESS_KEY=<secret-access-key>
```

Once those environment variables are set, run `azcopy` like this (replacing the AWS dataset location and `<blob_account_name>` with the name of your storage account:)

```shell
azcopy copy 'https://<AWS_region>.amazonaws.com/<AWS_dataset_name>' 'https://<blob_account_name>.blob.core.windows.net/datasets' --recursive=true
```

## Mount the Azure Storage Account

To inspect data or run preprocessing on the data, you need to make the data in the Azure storage account available on
your local workstation or on an Azure VM. This can be done by mounting the Azure Storage Account as a file system.

- Install `blobfuse` version 1 on your VM, following the instructions
  [in the docs](https://github.com/Azure/azure-storage-fuse/wiki/Blobfuse-Installation)
- Set two environment variables for the name of the
    storage account and the access key for the storage account. The access key can be found
    in the Azure Portal, under `Access keys` in the left-hand navigation bar. Then run in your `.bashrc` or in the terminal:

```bash
export AZURE_STORAGE_ACCOUNT=<name of the storage account>
export AZURE_STORAGE_ACCESS_KEY=<access key for the storage account>
```

- In the repository root, run `make mount_blobfuse`

You now have the `datasets` container of the storage account mounted at `/cyted`

If you're running into issues with mounting (`blobfuse` saying that it does not have the storage account name or key),
then

- Modify the file `setup/blobfuse.cfg` and set the `accountName` and `accountKey` variables to the right values
- Run `cd setup; ./mount_blob.sh` to mount the storage account

## Creating a datastore

A datastore is an abstraction in AzureML that allows you to access data in Azure Storage. It contains essentially the
name of the storage account and the access key. You can create a datastore in the AzureML workspace as follows:

- You need to know the name of the storage account and the access key for the storage account. The access key can be found
    in the Azure Portal, under `Access keys` in the left-hand navigation bar.
- In your AzureML workspace, find the "Data" section in the left-hand navigation bar
- Click "Datastores" and then "+ Create"
- Enter a name for the datastore, e.g. "cyted". We recommend to have the datastore name match the storage account name.
- Select "Datastore type: Azure Blob Storage"
- Choose "Account selection method: From Azure subscription" and select the subscription that contains the storage account.
- Choose the container that holds the data. If you followed the instructions here, this would be `datasets`.
- Select "Authentication type: Account key".
- Enter the access key for the storage account.

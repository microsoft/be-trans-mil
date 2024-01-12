#!/bin/bash
MOUNTPOINT=/cyted
sudo mkdir -p $MOUNTPOINT

# Install blobfuse using the instructions here: 
# https://learn.microsoft.com/en-us/azure/storage/blobs/storage-how-to-mount-container-linux
# Then fill in storage account name, container name and account key in blobfuse.cfg

sudo blobfuse $MOUNTPOINT \
    --tmp-path=/tmp/blobfuse \
    --config-file=blobfuse.cfg \
    -o attr_timeout=240 \
    -o entry_timeout=240 \
    -o negative_timeout=120 \
    -o allow_other
echo "Container mounted at $MOUNTPOINT"

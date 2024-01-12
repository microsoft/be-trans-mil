#!/bin/bash

# Create a blobfuse config file from the environment variables
cat << EOF > blobfuse.cfg
accountName $AZURE_STORAGE_ACCOUNT
accountKey $AZURE_STORAGE_KEY
containerName datasets
EOF

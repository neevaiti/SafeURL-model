#!/bin/bash

#### CHMOD FILES ####
for file in ../setup/azure_database/*.sh; do
  chmod +x "$file"
done

for file in ../setup/containers/*.sh; do
  chmod +x "$file"
done

echo "Files are now ready to use."
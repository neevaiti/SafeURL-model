#!/bin/bash

source ./.env


echo "Deleting database : $DB_NAME"
az postgres flexible-server db delete \
    --resource-group "$RESOURCE_GROUP" \
    --server-name "$SERVER_NAME" \
    --database-name "$DB_NAME"



echo "Deleting server : $SERVER_NAME"
az postgres flexible-server delete \
    --resource-group "$RESOURCE_GROUP" \
    --name "$SERVER_NAME"



echo "Deleting resource group : $RESOURCE_GROUP"
az group delete \
    --name "$RESOURCE_GROUP"
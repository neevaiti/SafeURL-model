#!/bin/bash

source ./.env

az postgres db delete \
    --resource-group $RESOURCE_GROUP \
    --server-name $SERVER_NAME \
    --name $DB_NAME


az postgres server delete \
    --resource-group $RESOURCE_GROUP \
    --name $SERVER_NAME


az group delete \
    --name $RESOURCE_GROUP
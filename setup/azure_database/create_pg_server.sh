#!/bin/bash

source ./.env

#### CREATE RESOURCE GROUP ####
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$SERVER_LOCATION"


#### CREATE SERVER ####
az postgres flexible-server create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$SERVER_NAME" \
    --location "$SERVER_LOCATION" \
    --admin-user "$ADMIN_USER" \
    --admin-password "$ADMIN_PASSWORD" \
    --tier "$TIER_SPECS" \
    --sku-name "$SKU_SPECS" \
    --version "$PG_VERSION" \
    --public "$PUBLIC_ACCESS" \
    --storage-size "$SERVER_STORAGE_SIZE"


#### CREATE DATABASE ####
az postgres db create \
    --resource-group $RESOURCE_GROUP \
    --server $SERVER_NAME \
    --name $DB_NAME
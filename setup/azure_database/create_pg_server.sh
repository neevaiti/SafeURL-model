#!/bin/bash

source ./.env

#### CREATE RESOURCE GROUP ####
echo "Creating resource group : $RESOURCE_GROUP"
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$SERVER_LOCATION"


#### CREATE SERVER ####
echo "Creating server : $SERVER_NAME"
az postgres flexible-server create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$SERVER_NAME" \
    --location "$SERVER_LOCATION" \
    --admin-user "$ADMIN_USER" \
    --admin-password "$ADMIN_PASS" \
    --tier "$TIER_SPECS" \
    --sku-name "$SKU_SPECS" \
    --version "$PG_VERSION" \
    --public "$PUBLIC_ACCESS" \
    --storage-size "$SERVER_STORAGE_SIZE"


#### CREATE DATABASE ####
echo "Creating database : $DB_NAME"
az postgres flexible-server db create \
    --resource-group "$RESOURCE_GROUP" \
    --server-name "$SERVER_NAME" \
    --database-name "$DB_NAME"
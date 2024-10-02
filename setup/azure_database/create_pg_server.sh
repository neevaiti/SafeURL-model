#!/bin/bash

source .env

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


#### FIND TABLES CREATOR SCRIPT ####
sqlScript="../azure_database/create_tables.sql"


#### CREATE DATABASE TABLES ####
psql -h "$SERVER_ADRESS" -d "$DB_NAME" -U "$ADMIN_USER" -f $sqlScript


#### ADD DATA TO List_url ####
listurlcsv="/Users/ant/Desktop/Projects/SafeURL-model/dataset/df_1000.csv"
psql -h "$SERVER_ADRESS" -d "$DB_NAME" -U "$ADMIN_USER" -c"\copy list_url FROM '$listurlcsv' DELIMITER ',' CSV HEADER;"
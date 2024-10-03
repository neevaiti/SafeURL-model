#!/bin/bash

source .env

#### FIND TABLES CREATOR SCRIPT ####
sqlScript="../azure_database/create_tables.sql"


#### CREATE DATABASE TABLES ####
psql -h "$SERVER_ADRESS" -d "$DB_NAME" -U "$ADMIN_USER" -f $sqlScript
#!/bin/bash

source .env


#### ADD DATA TO List_url ####
listurlcsv="/Users/ant/Desktop/Projects/SafeURL-model/dataset/df_all.csv"
psql -h "$SERVER_ADRESS" -d "$DB_NAME" -U "$ADMIN_USER" -c"\copy list_url FROM '$listurlcsv' DELIMITER ',' CSV HEADER;"
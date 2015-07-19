#!/bin/bash

# make sure current directory contains the unzipped "train" directory
# run the following: mkdir train_preprocessed; ./preprocess.sh
# script simply replaces "id" in the header with "subject,series,index"
# and replaces the id in subsequent rows with the corresponding values

for file in ./train/*.csv
do
    sed -e 's/_/,/g' -e 's/subj//' -e 's/series//'  -e '1s/id,/subject,series,index,/' $file >| "./train_preprocessed/$(basename $file)"
done

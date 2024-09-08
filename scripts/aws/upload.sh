#!/bin/bash

# Check if a file argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No file specified"
    echo "Usage: $0 <file_to_upload>"
    exit 1
fi

# Get the file path from the first argument
file_to_upload="$1"

# Check if the file exists
if [ ! -f "$file_to_upload" ]; then
    echo "Error: File '$file_to_upload' not found"
    exit 1
fi

# AWS S3 bucket name
bucket_name="genshot"
public_path="public"

# Upload the file to S3 with --no-progress and --delete options
aws s3 cp "$file_to_upload" "s3://$bucket_name/$public_path/"

# Check if the upload was successful
if [ $? -eq 0 ]; then
    echo "File '$file_to_upload' successfully uploaded to S3 bucket '$bucket_name'"
else
    echo "Error: Failed to upload file '$file_to_upload' to S3 bucket '$bucket_name'"
    exit 1
fi

#!/bin/bash

# Check if the user provided the sync interval and password
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <sync_interval_in_seconds> <password>"
  exit 1
fi

# Sync interval in seconds
SYNC_INTERVAL=$1

# SSH password
PASSWORD=$2

# Source directory on the local machine
LOCAL_DIR=~/pypi

# Destination directory on the HPC cluster
REMOTE_USER=your_username
REMOTE_HOST=hpc-login7
REMOTE_DIR=/path/to/destination/local_pypi

# Infinite loop to sync every specified interval
while true; do
  echo "Starting sync at $(date)"
  
  # Rsync command to sync local directory with remote directory using sshpass for password
  sshpass -p "$PASSWORD" rsync -avz --delete $LOCAL_DIR/ $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/
  
  echo "Sync completed at $(date)"
  
  # Wait for the specified interval
  sleep $SYNC_INTERVAL
done

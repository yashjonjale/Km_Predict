ls#!/bin/bash

# Check if the user provided the sync interval and password
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 <sync_interval_in_seconds> <password> <path>"
  exit 1
fi

# Sync interval in seconds
SYNC_INTERVAL=$1

# SSH password
PASSWORD=$2

# Source directory on the local machine
LOCAL_DIR=$3

# Destination directory on the HPC cluster
REMOTE_USER="not81yan"
REMOTE_HOST="storage.hpc.rz.uni-duesseldorf.de"
REMOTE_DIR="/gpfs/project/not81yan/${LOCAL_DIR}"

# Infinite loop to sync every specified interval
while true; do
  echo "Starting sync at $(date)"
  
  # Rsync command to sync local directory with remote directory using sshpass for password
  sshpass -p "$PASSWORD" rsync -avz $LOCAL_DIR/ $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/
  
  echo "Sync completed at $(date)"
  
  # Wait for the specified interval
  sleep $SYNC_INTERVAL
done

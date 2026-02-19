#!/bin/bash

# Cron job wrapper for granicus.py
# This script iterates through all subdomains and their video IDs

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APURVA_DIR="$SCRIPT_DIR/apurva"
cd "$APURVA_DIR"

# Configuration
SUBDOMAINS_FILE="$APURVA_DIR/data/subdomains.json"
STATE_DIR="$APURVA_DIR/logs/state"
LOG_FILE="$APURVA_DIR/logs/granicus_cron.log"
mkdir -p "$STATE_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Add timestamp to log
echo "=== Cron job started at $(date) ===" >> "$LOG_FILE" 2>&1

# Activate virtual environment if it exists
if [ -d "$APURVA_DIR/venv" ]; then
    source "$APURVA_DIR/venv/bin/activate"
    echo "Activated virtual environment" >> "$LOG_FILE" 2>&1
fi

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found" >> "$LOG_FILE" 2>&1
    exit 1
fi

# Check if jq is available for JSON parsing
if ! command -v jq &> /dev/null; then
    echo "ERROR: jq not found. Please install jq for JSON parsing." >> "$LOG_FILE" 2>&1
    echo "Install with: brew install jq" >> "$LOG_FILE" 2>&1
    exit 1
fi

# Extract all subdomain names from the JSON file
SUBDOMAINS=$(jq -r '.subdomains | keys[]' "$SUBDOMAINS_FILE")

# Iterate through each subdomain
for SUBDOMAIN in $SUBDOMAINS; do
    echo "Processing subdomain: $SUBDOMAIN" >> "$LOG_FILE" 2>&1
    
    STATE_FILE="$STATE_DIR/${SUBDOMAIN}_last_id.txt"
    
    # Read the last processed video ID for this subdomain, or start from 1
    if [ -f "$STATE_FILE" ]; then
        LAST_ID=$(cat "$STATE_FILE")
        CURRENT_ID=$((LAST_ID + 1))
    else
        CURRENT_ID=1
    fi
    
    echo "  Starting from video ID: $CURRENT_ID" >> "$LOG_FILE" 2>&1
    
    # Iterate through video IDs until one fails
    while true; do
        echo "  Processing $SUBDOMAIN video ID: $CURRENT_ID" >> "$LOG_FILE" 2>&1
        
        # Run granicus.py for the current video ID
        if python3 granicus.py "$SUBDOMAIN" "$CURRENT_ID" >> "$LOG_FILE" 2>&1; then
            echo "  Successfully processed $SUBDOMAIN video ID: $CURRENT_ID" >> "$LOG_FILE" 2>&1
            # Save the successful ID
            echo "$CURRENT_ID" > "$STATE_FILE"
            # Move to next ID
            CURRENT_ID=$((CURRENT_ID + 1))
        else
            echo "  Failed to process $SUBDOMAIN video ID: $CURRENT_ID - moving to next subdomain" >> "$LOG_FILE" 2>&1
            break
        fi
    done
done

echo "=== Cron job completed at $(date) ===" >> "$LOG_FILE" 2>&1

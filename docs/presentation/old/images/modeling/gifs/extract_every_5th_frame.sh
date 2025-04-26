#!/bin/bash

# Check if filename argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a GIF filename (e.g., expanding_window.gif)"
    echo "Usage: $0 <filename.gif>"
    exit 1
fi

# Get filename without extension
FILENAME=$(basename "$1" .gif)

# Define paths
INPUT_GIF="presentation/images/modeling/gifs/$1"
OUTPUT_DIR="presentation/images/modeling/gifs/extracted_frames"
FRAME_PATTERN="presentation/images/modeling/gifs/${FILENAME}_frame_%02d.png"

# Clean up any existing frame files
rm -f "presentation/images/modeling/gifs/${FILENAME}_frame_"*.png

# Step 1: Split the GIF into frames
convert -coalesce "$INPUT_GIF" "$FRAME_PATTERN"

# Step 2: Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 3: Extract every 5th frame
total_frames=$(ls "presentation/images/modeling/gifs/${FILENAME}_frame_"*.png | wc -l)
for i in $(seq 0 5 $((total_frames - 1))); do
    frame_number=$(printf "%02d" $i)
    cp "presentation/images/modeling/gifs/${FILENAME}_frame_${frame_number}.png" "$OUTPUT_DIR/${FILENAME}_frame_${frame_number}.png"
done

# Step 4: Clean up intermediate frames
rm -f "presentation/images/modeling/gifs/${FILENAME}_frame_"*.png

echo "Extraction complete. Every 5th frame is saved in '$OUTPUT_DIR'."

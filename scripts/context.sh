#!/bin/bash
ROOT_DIR="$(dirname "$0")"
OUTPUT_FILE="$ROOT_DIR/output/context.txt"
if [ -f "$OUTPUT_FILE" ]; then
  rm -f "$OUTPUT_FILE"
fi
echo "Below is a list of files for the https://github.com/hu-po/warp-ik codebase." >> "$OUTPUT_FILE"
process_file() {
  local file="$1"
  echo "Processing: $file"
  echo -e "\n\n--- BEGIN FILE: $file ---\n" >> "$OUTPUT_FILE"
  cat "$file" >> "$OUTPUT_FILE"
  echo -e "\n--- END FILE: $file ---\n" >> "$OUTPUT_FILE"
}
for specific_file in "README.md" ".env.example" "pyproject.toml"; do
  if [ -f "$specific_file" ]; then
    process_file "$specific_file"
  else
    echo "File not found: $specific_file"
  fi
done
declare -A DIRECTORIES=(
  ["docker"]=""
  ["src"]=""
  ["scripts"]=""
)
for dir in "${!DIRECTORIES[@]}"; do
  if [ -d "$dir" ]; then
    eval find "$dir" -type f -not -name "*.env" ${DIRECTORIES[$dir]} | while IFS= read -r file; do
      process_file "$file"
    done
  else
    echo "Directory not found: $dir"
  fi
done
echo -e "\n\n--- END OF CONTEXT ---\n" >> "$OUTPUT_FILE"
TOTAL_FILES=$(grep -c "^--- BEGIN FILE:" "$OUTPUT_FILE")
TOTAL_SIZE=$(du -h "$OUTPUT_FILE" | awk '{print $1}')
echo "Context file created at $OUTPUT_FILE"
echo "Total files: $TOTAL_FILES"
echo "Total size: $TOTAL_SIZE"
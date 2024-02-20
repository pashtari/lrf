#!/bin/bash

# Run Python script
python script.py

# Source the generated environment variables
source ckpt_path.sh

# Now you can use $VAR1 and $VAR2 in your bash script
echo "VAR is: $VAR"

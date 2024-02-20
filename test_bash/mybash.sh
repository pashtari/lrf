#!/bin/bash

# Run the first script and capture its output
output="$(python3 first_script.py)"

# Run the second script with modified arguments
python3 second_script.py arg2 ${output}  arg3

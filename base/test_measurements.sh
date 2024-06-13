#!/usr/bin/env bash
THIS_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
src=$THIS_SCRIPT_DIR"/results/pol/democratic_only.test.en"
tgt=$THIS_SCRIPT_DIR"/results/pol/democratic_only.test.to.republican.1"
echo $src
python3 measure_ts.py -src $src -tgt $tgt -type political 

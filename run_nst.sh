# DIR=$(realpath ".")
# DIR=$(dirname "${HERE}")

# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# echo "Script directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"


python "$SCRIPT_DIR/run_nst.py" fit \
    --data "$SCRIPT_DIR/config/data/nst/pacs.yaml" \
    --model "$SCRIPT_DIR/config/model/nst/ada_in.yaml" \
    --trainer.devices 1 \
    --trainer.accelerator gpu \
    --data.batch_size 8
# python /home/dycpu3_data1/yyalau/comp5331/run_nst.py fit \
#     --help

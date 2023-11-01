# DIR=$(realpath ".")
# DIR=$(dirname "${HERE}")

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo "Script directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"


python "$SCRIPT_DIR/run_fa.py" fit \
    --data "$SCRIPT_DIR/config/data/pacs.yaml" \
    --model "$SCRIPT_DIR/config/model/classification/fa/fagt.yaml" \
    --trainer.devices [2] \
    --trainer.accelerator cuda

# --data "$SCRIPT_DIR/config/data/office_home.yaml" \

# python /home/dycpu3_data1/yyalau/comp5331/run_nst.py fit \
#     --help

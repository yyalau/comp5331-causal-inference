DIR=$(realpath ".")
# DIR=$(dirname "${HERE}")

python "$DIR/run_erm.py" fit \
    --data "$DIR/config/data/digits.yaml" \
    --model "$DIR/config/model/classification/erm/vit.yaml" \
    --trainer.devices 1 \
    --trainer.accelerator gpu 
# python /home/dycpu3_data1/yyalau/comp5331/run_nst.py fit \
#     --help

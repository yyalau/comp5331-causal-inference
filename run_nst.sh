DIR=$(realpath ".")
# DIR=$(dirname "${HERE}")

python "$DIR/run_nst.py" fit \
    --data "$DIR/config/data/digits.yaml" \
    --model "$DIR/config/model/nst/ada_in.yaml" \
    --trainer.devices 1 \
    --trainer.accelerator gpu 
# python /home/dycpu3_data1/yyalau/comp5331/run_nst.py fit \
#     --help

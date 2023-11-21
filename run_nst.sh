SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SCRIPT_DIR"

python "$SCRIPT_DIR/run_nst.py" fit \
        --data "$SCRIPT_DIR/config/data/nst/pacs.yaml" \
        --model "$SCRIPT_DIR/config/model/nst/ada_in.yaml" \
        --trainer.devices [0] \
        --trainer.accelerator gpu \
        --trainer.max_epochs 70 \
        --data.batch_size 32 \
        --data.dataset_config.train_domains ["art_painting","cartoon","photo"] \
        --data.dataset_config.val_domains ["art_painting","cartoon","photo"] \
        --data.dataset_config.test_domains ["sketch"]

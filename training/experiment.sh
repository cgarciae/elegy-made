
python -m training.experiment \
    --epochs 1000 \
    --batch-size 256 \
    --n-units 1024 \
    --lr 0.001 \
    --n-components 10 \
    --a1 1.0 \
    --a2 0.0 \
    --n-layers 3 \
    --l2 0.0000 \
    "$@"

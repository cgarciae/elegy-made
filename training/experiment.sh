
python -m training.experiment \
    --epochs 1000 \
    --batch-size 64 \
    --n-units 256 \
    --lr 0.001 \
    --n-components 10 \
    --a1 1.0 \
    --a2 0.0 \
    --n-layers 3 \
    --l2 0.0005 \
    "$@"

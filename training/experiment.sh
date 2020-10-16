
python -m training.experiment \
    --epochs 1000 \
    --n-units 64 \
    --lr 0.001 \
    --n-components 15 \
    --a1 1.0 \
    --a2 0.01 \
    "$@"

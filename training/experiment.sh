
python -m training.experiment \
    --epochs 100 \
    --viz-steps 100 \
    --batch-size 256 \
    --n-units 2048 \
    --lr 0.005 \
    --n-components 10 \
    --a1 1.0 \
    --a2 0.01 \
    --n-layers 3 \
    --l2 0.0000 \
    --comp-red "sum" \
    "$@"

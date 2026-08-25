#!/bin/bash
# Train all WinRWKV model sizes sequentially

echo "Training all WinRWKV model sizes..."
echo "This will train: S, M, L, XL, XXL"
echo ""

MODEL_SIZES=("S" "M" "L" "XL" "XXL")
STEPS=750  # Medium training: 500-1000 steps

for size in "${MODEL_SIZES[@]}"; do
    echo "=========================================="
    echo "Training model size: $size"
    echo "=========================================="

    python train_winrwkv.py \
        --model_size "$size" \
        --steps $STEPS \
        --save_every 250 \
        --lr 1e-4

    if [ $? -ne 0 ]; then
        echo "Error training $size, continuing with next model..."
    fi

    echo ""
done

echo "Training complete!"

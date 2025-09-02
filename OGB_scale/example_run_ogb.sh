EPOCHS=100
BATCH=256
LR=1e-3
WD=1e-5
TEMP=0.2
PROJ=64
HIDDEN=32

mkdir -p "$BASE_DIR"

#ED - filling in the gaps.

for s in 1 2 3 4 5; do
  for PCT in 0.125 0.15 0.175; do
    OUTDIR="$BASE_DIR/ed_p0.10_pct${PCT}"
    mkdir -p "$OUTDIR"
    python graph_cl2.py \
      --result_dir "$OUTDIR" \
      --max_train_pct "$PCT" \
      --recipe ed --p 0.10 \
      --hidden_dim "$HIDDEN" --proj_dim "$PROJ" \
      --lr "$LR" --weight_decay "$WD" \
      --batch_size "$BATCH" --epochs "$EPOCHS" \
      --temp "$TEMP" --seed "$s"
  done
done

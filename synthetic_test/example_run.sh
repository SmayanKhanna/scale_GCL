EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-256}
DIM=${DIM:-32}
LAYERS=${LAYERS:-3}
PROJ=${PROJ:-2}
P=${P:-0.15}

DATA_ROOT=${DATA_ROOT:-data}
RESULTS_ROOT=${RESULTS_ROOT:-results_synth_2}

mkdir -p "$DATA_ROOT" "$RESULTS_ROOT"

for MULT in 2 4 6; do                      # style multipliers (graph size)
  for SIZE in 100 500 1000 2000; do        # per-class dataset size
    OUTDIR="$DATA_ROOT/synth6_S${MULT}_${SIZE}"
    TAG="S${MULT}_${SIZE}"

    # # # build once
    if [[ ! -f "$OUTDIR/data_list.pt" ]]; then
      echo "[build] $TAG"
      python build_multiclass_synth_TU.py \
        --samples_per_class "$SIZE" \
        --multiplier "$MULT" \
        --background_graph tree \
        --out_dir "$OUTDIR"
    else
      echo "[build] skip $TAG (exists)"
    fi

    # run seeds
    for SEED in 0 1 2; do
      echo "[run] GraphCL $TAG seed=$SEED"
      python infograph_synth.py \
        --synth_path "$OUTDIR" \
        --dataset_tag "$TAG" \
        --train_frac 1.0 \
        --epochs "$EPOCHS" --batch_size "$BATCH" \
        --dim "$DIM" --num_layers "$LAYERS" --proj_depth "$PROJ" \
        --seed "$SEED" \
        --result_dir "$RESULTS_ROOT"
    done
  done
done

#!/usr/bin/env bash

WORKDIR="/mnt/atlas02/users/leoli/software/pygama4scarf/src/prod"
SELF_METADATA="/mnt/atlas02/users/leoli/self_metadata"

DC="${SELF_METADATA}/data_cleaning.yaml"
MAPDIR="${SELF_METADATA}/vary_smoothing/maps_vcs"
SMOOTHING_LIST=("60" "100" "200")
MW_LIST=("1" "3" "5" "7")

# OUTPUT_BASE="/mnt/atlas02/users/leoli/scarf/"

# Dataset list: MODE PERIOD RUN
DATASETS=(
    "p00"
    "p05"
) 
echo "================================"
echo "super calibration production run."
echo "log summary will be saved to $LOGFILE"
echo "================================"
echo

cd "$WORKDIR" || {
    echo "Could not enter $WORKDIR"
    exit 1
}

LOGDIR="/mnt/atlas02/users/leoli/scarf/logs/scalib"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

echo "=== scalib production started $(date) ===" > "$LOGFILE"
for DATASET in "${DATASETS[@]}"; do
    read -r PERIOD <<< "$DATASET"
    LOGFILE="${LOGDIR}/production_log_${PERIOD}_${TIMESTAMP}.txt" 
    echo "================================"
    echo "Dataset:"
    echo "  Period:      $PERIOD"
    echo "================================"
    for S in "${SMOOTHING_LIST[@]}"; do
        for MW in "${MW_LIST[@]}"; do
            MAP="${MAPDIR}/map_calibration_s${S}ns_mw${MW}.yaml"
            RUNLOG="${LOGDIR}/scalib_s${S}ns_mw${MW}.log"
            echo "--------------------------------"
            echo "Would run:"
            printf 'calibrate_hpge_parallel.py \\\n'
            printf '    -p "%s" \\\n' "$PERIOD"
            if [[ "$PERIOD" == "p00" ]]; then
                printf '    -s "%s" \\\n' "p01"
            fi
            printf '    -m "%s" \\\n' "$MAP"
            printf '    -dc "%s" \\\n' "$DC"
            printf '    -n 6\n'
            if [[ "$PERIOD" == "p00" ]]; then
                supercalibrate_hpge_parallel.py \
                    -p "$PERIOD" \
                    -s "p01" \
                    -m "$MAP" \
                    -dc "$DC" \
                    -n 6 \
                    > "$RUNLOG" 2>&1
            else
                supercalibrate_hpge_parallel.py \
                    -p "$PERIOD" \
                    -m "$MAP" \
                    -dc "$DC" \
                    -n 6 \
                    > "$RUNLOG" 2>&1
            fi
            if [ $? -eq 0 ]; then
                echo "$(date '+%F %T') SUCCESS  ${PERIOD}/$ s${S}ns_mw${MW}" \
                    >> "$LOGFILE"
            else
                echo "$(date '+%F %T') FAILED  ${PERIOD}/$ s${S}ns_mw${MW}" \
                    >> "$LOGFILE"
            fi

        done
    done
done

echo "================================"
echo "Calibration dry run"
echo "Check the log files for details in $LOGDIR."
echo "================================"

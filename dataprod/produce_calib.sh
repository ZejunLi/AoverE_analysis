#!/usr/bin/env bash

WORKDIR="/mnt/atlas02/users/leoli/software/pygama4scarf/src/prod"
SELF_METADATA="/mnt/atlas02/users/leoli/self_metadata"

DC="${SELF_METADATA}/data_cleaning.yaml"


SMOOTHING_LIST=("300")
MW_LIST=("1" "3" "5" "7" "15")
# MW_LIST=("1")

OUTPUT_BASE="/mnt/atlas02/users/leoli/scarf/"

# Dataset list: MODE PERIOD RUN
DATASETS=(
    "cal p00 r037"
    "cal p05 r057"
)

echo "================================"
echo "Calibration production run."
echo "log summary will be saved to $LOGFILE"
echo "================================"
echo

cd "$WORKDIR" || {
    echo "Could not enter $WORKDIR"
    exit 1
}

LOGDIR="/mnt/atlas02/users/leoli/scarf/logs/calib"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="${LOGDIR}/production_log_${TIMESTAMP}.txt"

echo "=== DSP production started $(date) ===" > "$LOGFILE"
for DATASET in "${DATASETS[@]}"; do

    read -r MODE PERIOD RUN <<< "$DATASET"
    QC="/mnt/atlas01/users/vogl/scarf/sp01/metadata/hit/quality_v9-1/${PERIOD}/sp01-${PERIOD}-qc.yaml"
    echo "================================"
    echo "Dataset:"
    echo "  Mode:        $MODE"
    echo "  Period:      $PERIOD"
    echo "  Run:         $RUN"
    echo "  Input route: $IR"
    echo "================================"
    for S in "${SMOOTHING_LIST[@]}"; do
        for MW in "${MW_LIST[@]}"; do
            MAP="${SELF_METADATA}/vary_smoothing/maps_vcs/map_calibration_s${S}ns_mw${MW}.yaml"
            CONF="${SELF_METADATA}/vary_smoothing/maps_vcs/ged_dsp_conf_s${S}ns_mw${MW}.yaml"
            OR="${OUTPUT_BASE}/metadata/calib/s${S}ns_mw${MW}"
            RUNLOG="${LOGDIR}/${MODE}_${PERIOD}_${RUN}_s${S}ns_mw${MW}.log"
            echo "--------------------------------"
            echo "Would run:"
            printf 'calibrate_hpge_parallel.py \\\n'
            printf '    -e "%s" \\\n' "sp01"
            printf '    -p "%s" \\\n' "$PERIOD"
            printf '    -d "%s" \\\n' "$MODE"
            printf '    -o "%s"\n' "$OR"
            printf '    -dc "%s" \\\n' "$DC"
            printf '    -m "%s" \\\n' "$MAP"
            printf '    -qc "%s" \\\n' "$QC"
            printf '    -n 4\n'
            calibrate_hpge_parallel.py \
                -e "sp01" \
                -p "$PERIOD"\
                -d "$MODE" \
                -o "$OR"\
                -dc "$DC"\
                -m "$MAP"\
                -qc "$QC"\
                -n 2\
                > "$RUNLOG" 2>&1
            if [ $? -eq 0 ]; then
                echo "$(date '+%F %T') SUCCESS ${MODE}/${PERIOD}/${RUN} s${S}ns_mw${MW}" \
                    >> "$LOGFILE"
            else
                echo "$(date '+%F %T') FAILED  ${MODE}/${PERIOD}/${RUN} s${S}ns_mw${MW}" \
                    >> "$LOGFILE"
            fi

        done
    done
done

echo "================================"
echo "Calibration run"
echo "Check the log files for details in $LOGDIR."
echo "================================"
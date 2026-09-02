#!/usr/bin/env bash

WORKDIR="/mnt/atlas02/users/leoli/software/pygama4scarf/src/prod"
SELF_METADATA="/mnt/atlas02/users/leoli/self_metadata"
MAPdir="${SELF_METADATA}/vary_smoothing/maps_vcs"

# SMOOTHING_LIST=("30")
# MW_LIST=("5")
SMOOTHING_LIST=("30" "40" "50" "300" "500" "700")
MW_LIST=("1" "3" "5" "7")
OUTPUT_BASE="/mnt/atlas02/users/leoli/scarf/sp01/data/sit_vcs"
scalib_BASE="/mnt/atlas02/users/leoli/scarf/sp01/metadata/scalib"
# Dataset list: MODE PERIOD RUN
DATASETS=(
    "cal p00 r037"
    "phy p00 r038"
    "cal p05 r057"
    "phy p05 r059"
    'cal p01 r040'
    'cal p01 r042'
    'cal p01 r044'
    'cal p05 r061'
    'cal p05 r064'
    'phy p05 r060'
) 

echo "================================"
echo "SIT production run."
echo "log summary will be saved to $LOGFILE"
echo "================================"
echo

cd "$WORKDIR" || {
    echo "Could not enter $WORKDIR"
    exit 1
}

LOGDIR="/mnt/atlas02/users/leoli/scarf/logs/sit_vcs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="${LOGDIR}/production_log_${TIMESTAMP}.txt"

echo "=== SIT production started $(date) ===" > "$LOGFILE"
for DATASET in "${DATASETS[@]}"; do
    read -r MODE PERIOD RUN <<< "$DATASET"
    echo "================================"
    echo "Dataset:"
    echo "  Mode:        $MODE"
    echo "  Period:      $PERIOD"
    echo "  Run:         $RUN"
    echo "  Input route: $IR"
    echo "================================"

    for S in "${SMOOTHING_LIST[@]}"; do
        for MW in "${MW_LIST[@]}"; do
            MAP="${MAPdir}/map_calibration_s${S}ns_mw${MW}.yaml"
            IR="/mnt/atlas02/users/leoli/scarf/sp01/data/hit_vcs/s${S}ns_mw${MW}"
            OR="${OUTPUT_BASE}/s${S}ns_mw${MW}/"
            SCALIB="${scalib_BASE}/s${S}ns_mw${MW}/${PERIOD}/${PERIOD}-AoE_supercal.yaml"
            CONF="${SELF_METADATA}/operations_sit_ged.yaml"
            RUNLOG="${LOGDIR}/${MODE}_${PERIOD}_${RUN}_s${S}ns_mw${MW}.log"
            echo "--------------------------------"
            echo "Would run:"
            printf './produce_sit.py \\\n'
            printf '    -p "%s" \\\n' "$PERIOD"
            printf '    -r "%s" \\\n' "$RUN"
            printf '    -d "%s" \\\n' "$MODE"
            printf '    -ged "%s" \\\n' "${SCALIB}"
            printf '    -c "%s" \\\n' "$CONF"
            printf '    -m "%s" \\\n' "$MAP"
            printf '    -i "%s" \\\n' "$IR"
            printf '    -o "%s"\n' "$OR"
            printf '    -n 4\n'
            produce_sit.py \
                -p "$PERIOD" \
                -r "$RUN" \
                -d "$MODE" \
                -ged "$SCALIB" \
                -c "$CONF" \
                -m "$MAP" \
                -i "$IR" \
                -o "$OR" \
                -n 4 \
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
echo "SIT production finished."
echo "Check the log files for details in $LOGDIR."
echo "================================"


# options:
#   -h, --help            show this help message and exit
#   -p, --period PERIOD   Period identifier, e.g. p02.
#   -r, --run RUN         Run identifier, e.g. r049.
#   -d, --datatype {cal,phy}
#                         Data type to process.
#   -c, --config CONFIG   Path to build hit configuration file containing outputs and operations. Also called operations file.
#   -ged, --ged-calibration GED_CALIBRATION
#                         Path to HPGe supercalibration file.
#   -i, --in_base_dir IN_BASE_DIR
#                         Optional. Base directory for input HIT tier data. If not provided, default from map is used.
#   -o, --out_base_dir OUT_BASE_DIR
#                         Optional. Base directory for output SIT data. If not provided, default from map is used.
#   -m, --map MAP         Map. Defines which LH5 group contains which detector and many other things.
#   -n, --n_workers N_WORKERS
#                         Optional. Maximal number of workers (i.e., CPU cores).
#   --skip-existing       Optional. Skip output files that already exist instead of raising an error.
#   --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
#                         Logging level (default: INFO)
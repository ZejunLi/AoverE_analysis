#!/usr/bin/env bash

WORKDIR="/mnt/atlas02/users/leoli/software/pygama4scarf/src/prod"
SELF_METADATA="/mnt/atlas02/users/leoli/self_metadata"

DB="${SELF_METADATA}/dsp_database_general.yaml"
MAP="${SELF_METADATA}/map.yaml"

SMOOTHING_LIST=("30")
MW_LIST=("3")
# SMOOTHING_LIST=("30" "40" "50" "300" "500" "700")
# MW_LIST=("1" "3" "5" "7")
OUTPUT_BASE="/mnt/atlas02/users/leoli/scarf/sp01/data/dsp_vcs"

# Dataset list: MODE PERIOD RUN
DATASETS=(
    # "cal p00 r037"
    # "phy p00 r038"
    # "cal p05 r057"
    # "phy p05 r059"
    # 'cal p01 r040'
    # 'cal p01 r042'
    # 'cal p01 r044'
    # 'cal p05 r061'
    'cal p05 r064'
    # 'phy p05 r060'
) 

echo "================================"
echo "DSP production run."
echo "log summary will be saved to $LOGFILE"
echo "================================"
echo

cd "$WORKDIR" || {
    echo "Could not enter $WORKDIR"
    exit 1
}

LOGDIR="/mnt/atlas02/users/leoli/scarf/logs/dsp_vcs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="${LOGDIR}/production_log_${TIMESTAMP}.txt"

echo "=== DSP production started $(date) ===" > "$LOGFILE"
for DATASET in "${DATASETS[@]}"; do

    read -r MODE PERIOD RUN <<< "$DATASET"

    IR="/mnt/atlas01/projects/scarf/data/sp01/prodenv/ref-raw/generated/tier/raw/${MODE}/${PERIOD}/${RUN}"

    echo "================================"
    echo "Dataset:"
    echo "  Mode:        $MODE"
    echo "  Period:      $PERIOD"
    echo "  Run:         $RUN"
    echo "  Input route: $IR"
    echo "================================"

    for S in "${SMOOTHING_LIST[@]}"; do
        for MW in "${MW_LIST[@]}"; do

            CONF="${SELF_METADATA}/vary_smoothing/dsp_vcs/ged_dsp_conf_s${S}ns_mw${MW}.yaml"
            OR="${OUTPUT_BASE}/s${S}ns_mw${MW}/${MODE}/${PERIOD}/${RUN}"
            RUNLOG="${LOGDIR}/${MODE}_${PERIOD}_${RUN}_s${S}ns_mw${MW}.log"
            echo "--------------------------------"
            echo "Would run:"
            printf './produce_dsp.py \\\n'
            printf '    -ged "%s" \\\n' "$CONF"
            printf '    -db "%s" \\\n' "$DB"
            printf '    -m "%s" \\\n' "$MAP"
            printf '    -ir "%s" \\\n' "$IR"
            printf '    -or "%s"\n' "$OR"
            printf '    -n 4\n'
            produce_dsp.py \
                -ged "$CONF" \
                -db "$DB" \
                -m "$MAP" \
                -ir "$IR" \
                -or "$OR" \
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
echo "DSP production finished."
echo "Check the log files for details in $LOGDIR."
echo "================================"
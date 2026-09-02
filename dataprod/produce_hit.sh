#!/usr/bin/env bash

WORKDIR="/mnt/atlas02/users/leoli/software/pygama4scarf/src/prod"
SELF_METADATA="/mnt/atlas02/users/leoli/self_metadata"

DB="${SELF_METADATA}/dsp_database_general.yaml"
MAP="${SELF_METADATA}/map.yaml"

# SMOOTHING_LIST=("300" "500" "700" "1000" "2000")
SMOOTHING_LIST=("300" "500" "700")
MW_LIST=("1" "3" "5" "7")

# Dataset list: MODE PERIOD RUN
DATASETS=(
    # "phy p00 r038"
    # "cal p05 r057"
    # "phy p05 r059"
    'cal p01 r040'
    'cal p01 r042'
    'cal p01 r044'
    'cal p05 r061'
    'phy p05 r060'
)

echo "================================"
echo "hit production run."
echo "log summary will be saved to $LOGFILE"
echo "================================"
echo

cd "$WORKDIR" || {
    echo "Could not enter $WORKDIR"
    exit 1
}

LOGDIR="/mnt/atlas02/users/leoli/scarf/logs/hit_vcs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="${LOGDIR}/production_log_${TIMESTAMP}.txt"
CONFIGFILE='/mnt/atlas02/users/leoli/self_metadata/operations_hit_ged.yaml'
echo "=== DSP production started $(date) ===" > "$LOGFILE"
for DATASET in "${DATASETS[@]}"; do

    read -r Datatype PERIOD RUN <<< "$DATASET"

    echo "================================"
    echo "Dataset:"
    echo "  Datatype:    $Datatype"
    echo "  Period:      $PERIOD"
    echo "  Run:         $RUN"
    echo "  Input route: $IR"
    echo "================================"

    QC="/mnt/atlas01/users/vogl/scarf/sp01/metadata/hit/quality_v9-1/${PERIOD}/sp01-${PERIOD}-qc.yaml"

    for S in "${SMOOTHING_LIST[@]}"; do
        for MW in "${MW_LIST[@]}"; do
            if [[ "$RUN" == "r060" ]]; then
                Cal_RUN="r057"
            else 
                Cal_RUN="$RUN"
            fi

            echo "RUN=$RUN, CAL_RUN=$CAL_RUN, S=$S, MW=$MW"
            CALIB="/mnt/atlas02/users/leoli/scarf/metadata/calib/s${S}ns_mw${MW}/cal/${PERIOD}/${Cal_RUN}/sp01-${PERIOD}-${Cal_RUN}-cal-hpge-calib.yaml"
            RUNLOG="${LOGDIR}/${Datatype}_${PERIOD}_${RUN}_s${S}ns_mw${MW}.log"
            MAP="${SELF_METADATA}/vary_smoothing/maps_vcs/map_calibration_s${S}ns_mw${MW}.yaml"

            cmd=(
                produce_hit.py
                -r "$RUN"
                -ged "$CALIB"
                -p "$PERIOD"
                -d "$Datatype"
                -m "$MAP"
                -c "$CONFIGFILE"
            )

            if [[ "$Datatype" == "phy" ]]; then
                cmd+=(-t CAL_TO_PHY)
            elif [[ "$Datatype" != "cal" ]]; then
                echo "Unknown datatype: $Datatype"
                continue
            fi

            cmd+=(
                -qc "$QC"
                -n 4
            )

            echo "--------------------------------"
            echo "Would run:"
            printf ' %q' "${cmd[@]}"
            printf '\n'

            if "${cmd[@]}" > "$RUNLOG" 2>&1; then
                echo "$(date '+%F %T') SUCCESS ${Datatype}/${PERIOD}/${RUN} s${S}ns_mw${MW}" \
                    >> "$LOGFILE"
            else
                echo "$(date '+%F %T') FAILED  ${Datatype}/${PERIOD}/${RUN} s${S}ns_mw${MW}" \
                    >> "$LOGFILE"
            fi

        done
    done
done


echo "================================"
echo "hit produced."
echo "Check the log files for details in $LOGDIR."
echo "================================"



# produce_hit.py 
# -p p00 
# -d cal 
# -c /mnt/atlas02/users/leoli/self_metadata/operations_hit_ged.yaml 
# -ged /mnt/atlas02/users/leoli/scarf/sp01/metadata/calib/s300ns_mw1/cal/p00/r037/sp01-p00-r037-cal-hpge-calib.yaml 
# -qc /mnt/atlas01/users/vogl/scarf/sp01/metadata/hit/quality_v9-1/p00/sp01-p00-qc.yaml 
# -m /mnt/atlas02/users/leoli/self_metadata/map_calibration_s300ns_mw1.yaml -n 4
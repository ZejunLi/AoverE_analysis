import lh5
import numpy as np
import awkward as ak
import os
import re
import gc
import pickle
import numpy as np
from pathlib import Path
filepath = "/mnt/atlas02/users/leoli/scarf/sp01/data/hit_vcs"
files = os.listdir(filepath)
files.sort()
files

pnum_cal = "00"
run_cal = "037"

pnum_phy_bg = "00"
run_phy_bg = "038"

pnum_phy_signal = "05"
run_phy_signal = "059"

runs = [
    # 's100ns_mw1', 's100ns_mw3', 's100ns_mw5', 's100ns_mw7',
    # 's200ns_mw1', 's200ns_mw3', 's200ns_mw5', 's200ns_mw7',
    # 's60ns_mw1',  's60ns_mw3',  's60ns_mw5',  's60ns_mw7',
    # 's300ns_mw1', 's300ns_mw3', 's300ns_mw5', 's300ns_mw7','s300ns_mw15',
    # 's500ns_mw1', 's500ns_mw3', 's500ns_mw5', 's500ns_mw7','s500ns_mw15',
    # 's700ns_mw1', 's700ns_mw3', 's700ns_mw5', 's700ns_mw7','s700ns_mw15',
    # 's1000ns_mw1',  's1000ns_mw3',  's1000ns_mw5',  's1000ns_mw7','s1000ns_mw15',
    's2000ns_mw3',  's2000ns_mw5',  's2000ns_mw7',
]

def build_run_db(runs, filepath):
    db = {}
    missing = []

    for run in runs:
        m = re.match(r"s(\d+)ns_mw(\d+)", run)
        if not m:
            missing.append(f"{run}: invalid run name format")
            continue

        run_path = os.path.join(filepath, run)
        phy_bg_path = os.path.join(run_path, "phy", f"p{pnum_phy_bg}", f"r{run_phy_bg}")
        phy_signal_path = os.path.join(run_path, "phy", f"p{pnum_phy_signal}", f"r{run_phy_signal}")
        cal_path = os.path.join(run_path, "cal", f"p{pnum_cal}", f"r{run_cal}")

        missing_this_run = []

        if not os.path.exists(run_path):
            missing_this_run.append(f"run folder missing: {run_path}")
        if not os.path.exists(phy_bg_path):
            missing_this_run.append(f"phy_bg path missing: {phy_bg_path}")
        if not os.path.exists(phy_signal_path):
            missing_this_run.append(f"phy_signal path missing: {phy_signal_path}")
        if not os.path.exists(cal_path):
            missing_this_run.append(f"cal path missing: {cal_path}")

        if missing_this_run:
            missing.append(f"{run}\n  " + "\n  ".join(missing_this_run))
            continue

        phy_bg_files = sorted(os.listdir(phy_bg_path))
        phy_signal_files = sorted(os.listdir(phy_signal_path))
        cal_files = sorted(os.listdir(cal_path))

        db[run] = {
            "run": run,
            "interval": f"{m.group(1)}ns",
            "mw": f"mw{m.group(2)}",

            "phy_bg_path": phy_bg_path,
            "phy_signal_path": phy_signal_path,
            "cal_path": cal_path,

            "phy_bg_files": phy_bg_files,
            "phy_signal_files": phy_signal_files,
            "cal_files": cal_files,

            "phy_bg_pathlist": [os.path.join(phy_bg_path, f) for f in phy_bg_files],
            "phy_signal_pathlist": [os.path.join(phy_signal_path, f) for f in phy_signal_files],
            "cal_pathlist": [os.path.join(cal_path, f) for f in cal_files],
        }

    if missing:
        raise FileNotFoundError(
            "[ERROR] Some runs or paths are missing:\n\n" + "\n\n".join(missing)
        )

    return db
run_db = build_run_db(runs, filepath)
output_dir = Path("AOE_classifier_pickles")
output_dir.mkdir(exist_ok=True)

for run in runs:
    print(f"Processing {run}...")

    classifier = np.asarray(
        lh5.read(
            "ch002/hit/A_max_mw_o_E_Classifier",
            run_db[run]["phy_signal_pathlist"],
        )
    )

    low_cut = np.asarray(
        lh5.read(
            "ch002/hit/A_max_mw_o_E_Low_Cut",
            run_db[run]["phy_signal_pathlist"],
        )
    )
    E = np.asarray(
        lh5.read(
            "ch002/hit/trapEftp_fix_cal",
            run_db[run]["phy_signal_pathlist"],
        )
    )
    # Use this if ROI_mask is a dictionary indexed by run
    # roi_mask_run = np.asarray(ROI_mask[run])

    obj = {
        "A_max_mw_o_E_Classifier": classifier,
        "A_max_mw_o_E_Low_Cut": low_cut,
        "trapEftp_fix_cal":E,
    }

    output_file = output_dir / f"AOE_Classifier_r059_{run}.pkl"

    with open(output_file, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {output_file}")

    # Remove all references to this run's arrays
    del obj
    del classifier
    del low_cut
    del E
    
    gc.collect()
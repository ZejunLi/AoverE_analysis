# Some ideas to do the cut better.
## first 
# %% [markdown]
# # Main heading
#
# This is normal **Markdown** text.
#
# - First item
# - Second item
#
# Equation:
# $$
# E = mc^2
# $$

# %%
from pytools4scarf import dataloader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
# %%
from pathlib import Path
runs = [
    's20ns_mw5',  's20ns_mw7',
    's30ns_mw3',  's30ns_mw5',  's30ns_mw7',
    's40ns_mw1',  's40ns_mw3',  's40ns_mw5',  's40ns_mw7',
    's50ns_mw1',  's50ns_mw3',  's50ns_mw5',  's50ns_mw7',
    's60ns_mw1',  's60ns_mw3',  's60ns_mw5',  's60ns_mw7',
    's100ns_mw1', 's100ns_mw3', 's100ns_mw5', 's100ns_mw7',
    's200ns_mw1', 's200ns_mw3', 's200ns_mw5', 's200ns_mw7',
    's300ns_mw1', 's300ns_mw3', 's300ns_mw5', 's300ns_mw7','s300ns_mw15',
    's500ns_mw1', 's500ns_mw3', 's500ns_mw5', 's500ns_mw7','s500ns_mw15',
    's700ns_mw1', 's700ns_mw3', 's700ns_mw5', 's700ns_mw7','s700ns_mw15',
]

map_dir = Path(
    "/mnt/atlas02/users/leoli/self_metadata/vary_smoothing/maps_vcs"
)
def map_path(config):
    map_dir ="/mnt/atlas02/users/leoli/self_metadata/vary_smoothing/maps_vcs"
    return map_dir+"/map_calibration_"+config+".yaml"
data_cleaning_path = (
    "/mnt/atlas02/users/leoli/self_metadata/data_cleaning.yaml"
)
# %%
run_1 = "s40ns_mw7"
run_2 = "s500ns_mw3"
# %%
dfs_evt,lts_evt = {},{}
dfs_evt["r059"], lts_evt["r059"] = dataloader.load_lh5_run(
    ["ch002/evt"],
    "/mnt/atlas01/users/vogl/scarf/sp01/metadata/map.yaml",
    data_cleaning_path,
    "evt",
    "sp01",
    "p05",
    "r059",
    "phy",
    ["ch002"],
)
# %%
dfs_phy_evt = dfs_evt['r059']['ch002']
argon_veto = np.asarray(dfs_phy_evt.is_lar_vetoed)
muon_veto = np.asarray(dfs_phy_evt.is_muon)
alpha_veto = np.asarray(dfs_phy_evt.A_max_o_E_High_Side_Cut & (dfs_phy_evt.Iasy<0))
# %%
dfs_1, lts_1 = {}, {}
dfs_1["r059"], lts_1["r059"] = dataloader.load_lh5_run(
    ["ch002/hit"],
    map_path(run_1),
    data_cleaning_path,
    "sit",
    "sp01",
    "p05",
    "r059",
    "phy",
    ["ch002"],
)
dfs_2, lts_2 = {}, {}
dfs_2["r059"], lts_2["r059"] = dataloader.load_lh5_run(
    ["ch002/hit"],
    map_path(run_2),
    data_cleaning_path,
    "sit",
    "sp01",
    "p05",
    "r059",
    "phy",
    ["ch002"],
)
# %%
dfs_phy_1 = dfs_1['r059']['ch002']
dfs_phy_2 = dfs_2['r059']['ch002']
# %%
E_1=np.asarray(dfs_phy_1.trapEmax_fix_cal)
A_max_mw_o_E_Classifier_1 = np.asarray(dfs_phy_1.A_max_mw_o_E_fix_Classifier)
is_valid_waveform_1 = np.asarray(dfs_phy_1.is_valid_waveform)
E_2=np.asarray(dfs_phy_2.trapEmax_fix_cal)
A_max_mw_o_E_Classifier_2 = np.asarray(dfs_phy_2.A_max_mw_o_E_fix_Classifier)
is_valid_waveform_2 = np.asarray(dfs_phy_2.is_valid_waveform)
# %%
import pickle 
with open("/mnt/atlas02/users/leoli/AoverE_analysis/both_sided_cuts/survival_db_from_dataprod.pkl", "rb") as f:
    survival_db = pickle.load(f)
ROI_mask = (E_1 > 1839) & (E_1 < 2239)
# ROI_mask = (E_2 > 1839) & (E_2 < 2239)

cut_value_1 = survival_db[run_1]['cut_value_90']
cut_value_2 = survival_db[run_2]['cut_value_90']
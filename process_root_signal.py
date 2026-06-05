import uproot
import awkward as ak # for data manipulation
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt # common shorthand
from pathlib import Path


# In Dir 
in_dir = '/Users/mghrear/data/LDMX_GNN/ldmx-det-v14-8gev_no_filter_2/signal_Ap1.0GeV_1e/'
# Out Dir
out_dir = '/Users/mghrear/data/LDMX_GNN/ldmx-det-v14-8gev_no_filter_2/signal_Ap1.0GeV_1e_processed/'

passName="signal" # "TrackerReco", "signal"

#################################################################################


def get_root_files(directory):
    """
    Get a list of .root files in the specified directory without the .root extension.
    
    Parameters:
    -----------
    directory : str or Path
        Path to the directory to search
        
    Returns:
    --------
    list
        List of .root filenames without the .root extension
    """
    dir_path = Path(directory)
    
    # Check if directory exists
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    
    # Get all .root files and remove the .root extension
    root_files = [f.stem for f in dir_path.glob("*.root")]
    
    return sorted(root_files)  # Return sorted list for consistency

file_names = get_root_files(in_dir)


for file_name in file_names:

    # Root file location and ldmx-sw pass name
    root_file =uproot.open(in_dir+file_name+'.root')['LDMX_Events']


    # Laod in data arrays from root file

    # Load SimParticle Info
    SimParticles_first = root_file['SimParticles_'+passName+'/SimParticles_'+passName+'.first'].array()
    SimParticles_processType = root_file['SimParticles_'+passName+'/SimParticles_'+passName+'.second.processType_'].array()
    SimParticles_pdgID = root_file['SimParticles_'+passName+'/SimParticles_'+passName+'.second.pdgID_'].array()
    SimParticles_parents = root_file['SimParticles_'+passName+'/SimParticles_'+passName+'.second.parents_'].array()
    SimParticles_px = root_file['SimParticles_'+passName+'/SimParticles_'+passName+'.second.px_'].array()
    SimParticles_py = root_file['SimParticles_'+passName+'/SimParticles_'+passName+'.second.py_'].array()
    SimParticles_pz = root_file['SimParticles_'+passName+'/SimParticles_'+passName+'.second.pz_'].array()


    # Load Tagger info
    Tagger_SimHits_x = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.x_'].array()
    Tagger_SimHits_y = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.y_'].array()
    Tagger_SimHits_z = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.z_'].array()
    Tagger_SimHits_pdgID = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.pdgID_'].array()
    Tagger_SimHits_Edep = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.edep_'].array()
    Tagger_SimHits_trackID = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.trackID_'].array()

    Tagger_Digi_x = root_file['DigiTaggerSimHits_'+passName+'/DigiTaggerSimHits_'+passName+'.x_'].array()
    Tagger_Digi_y = root_file['DigiTaggerSimHits_'+passName+'/DigiTaggerSimHits_'+passName+'.y_'].array()
    Tagger_Digi_z = root_file['DigiTaggerSimHits_'+passName+'/DigiTaggerSimHits_'+passName+'.z_'].array()
    Tagger_Digi_Edep = root_file['DigiTaggerSimHits_'+passName+'/DigiTaggerSimHits_'+passName+'.edep_'].array() 
    Tagger_Digi_trackID = root_file['DigiTaggerSimHits_'+passName+'/DigiTaggerSimHits_'+passName+'.trackIds_'].array()

    Tagger_Truth_ID = root_file['TaggerTruthTracks_'+passName+'/TaggerTruthTracks_'+passName+'.trackID_'].array()
    Tagger_Truth_P = (1.0/root_file['TaggerTruthTracks_'+passName+'/TaggerTruthTracks_'+passName+'.perigee_pars_'].array())[:,:,4]

    # Load Recoil info
    Recoil_SimHits_x = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.x_'].array()
    Recoil_SimHits_y = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.y_'].array()
    Recoil_SimHits_z = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.z_'].array()
    Recoil_SimHits_pdgID = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.pdgID_'].array()
    Recoil_SimHits_Edep = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.edep_'].array()
    Recoil_SimHits_trackID = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.trackID_'].array()

    Recoil_Digi_x = root_file['DigiRecoilSimHits_'+passName+'/DigiRecoilSimHits_'+passName+'.x_'].array()
    Recoil_Digi_y = root_file['DigiRecoilSimHits_'+passName+'/DigiRecoilSimHits_'+passName+'.y_'].array()
    Recoil_Digi_z = root_file['DigiRecoilSimHits_'+passName+'/DigiRecoilSimHits_'+passName+'.z_'].array()
    Recoil_Digi_Edep = root_file['DigiRecoilSimHits_'+passName+'/DigiRecoilSimHits_'+passName+'.edep_'].array() 
    Recoil_Digi_trackID = root_file['DigiRecoilSimHits_'+passName+'/DigiRecoilSimHits_'+passName+'.trackIds_'].array()

    Recoil_Truth_ID = root_file['RecoilTruthTracks_'+passName+'/RecoilTruthTracks_'+passName+'.trackID_'].array()
    Recoil_Truth_P = (1.0/root_file['RecoilTruthTracks_'+passName+'/RecoilTruthTracks_'+passName+'.perigee_pars_'].array())[:,:,4]

    Process_Type = root_file['SimParticles_'+passName+'/SimParticles_'+passName+'.second.process_type_'].array()

    rows = []
    for i in range(len(Tagger_SimHits_x)):
        rows.append({"SimParticles_first": np.array(SimParticles_first[i]),
                     "SimParticles_n_primaries": np.sum(np.array(Process_Type[i]) == 13),
                     "SimParticles_processType": np.array(SimParticles_processType[i]), 
                     "SimParticles_pdgID": np.array(SimParticles_pdgID[i]), 
                     "SimParticles_parents": np.array(SimParticles_parents[i]).flatten(), 
                     "SimParticles_px": np.array(SimParticles_px[i]),
                     "SimParticles_py": np.array(SimParticles_py[i]), 
                     "SimParticles_pz": np.array(SimParticles_pz[i]), 
                     "Tagger_Digi_x": np.array(Tagger_Digi_x[i]),
                     "Tagger_Digi_y": np.array(Tagger_Digi_y[i]), 
                     "Tagger_Digi_z": np.array(Tagger_Digi_z[i]), 
                     "Tagger_Digi_Edep": np.array(Tagger_Digi_Edep[i]), 
                     "Tagger_Digi_trackID": np.array(Tagger_Digi_trackID[i]).flatten(), 
                     "Tagger_TruthID": np.array(Tagger_Truth_ID[i]), 
                     "Tagger_TruthP": np.abs(np.array(Tagger_Truth_P[i])), 
                     "Recoil_Digi_x": np.array(Recoil_Digi_x[i]), 
                     "Recoil_Digi_y": np.array(Recoil_Digi_y[i]), 
                     "Recoil_Digi_z": np.array(Recoil_Digi_z[i]), 
                     "Recoil_Digi_Edep": np.array(Recoil_Digi_Edep[i]), 
                     "Recoil_Digi_trackID": np.array(Recoil_Digi_trackID[i]).flatten(), 
                     "Recoil_TruthID": np.array(Recoil_Truth_ID[i]), 
                     "Recoil_TruthP": np.abs(np.array(Recoil_Truth_P[i])) })

    df = pd.DataFrame(rows)
    df.to_pickle(out_dir+file_name+'.pkl')

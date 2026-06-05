# GAIN
Graph attention interaction network (GAIN) implementation via Pytorch Geometric. Notebooks in this repository are written such that they can be run on Google Collab, only the the filepaths need to be updated. Data is available upon request.

Below is a list of the included scripts, in order of dependency

## models.py
This contains a PyG implementation of GAIN layer as well as several GAIN and Interaction Network (IN) models.

## GNNTrackingTools.py
This contains tools used throughout. Most of these are LDMX-specifc but can easily be adapted.

## process_root_inclusive.py and process_root_signal.py
These scripts are used to process the raw root files for the inclusive (multi-electron tagger) and signal (recoil tracker) scenarios investigate. This is mainly done so that the PyG environment used does not need ROOT.

## LDMX_GAIN_tagger.ipynb and LDMX_IN_tagger.ipynb
Notebooks in which the nominal GAIN and IN models are trained and tested for the multi-electron tagging scenario. The _5L notebooks showcase the larger 5 message-passing layer models.

## LDMX_GAIN_signal_train.ipynb and LDMX_IN_signal_train.ipynb 
Notebooks in which the nominal GAIN and IN models are trained for the signal recoil electron tracking scenario.

## LDMX_GAIN_signal_test.ipynb and LDMX_IN_signal_test.ipynb 
Notebooks in which the nominal GAIN and IN models are tested for the signal recoil electron tracking scenario.

## make_figs.ipynb
Uses the results from the previous notebooks to make the main figures included in the GAIN paper.

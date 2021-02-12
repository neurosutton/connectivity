# Exploring graph theory connectivity of neuroimaging data
Produces graph theory summaries based on connectivity matrices generated with the CONN toolbox.

For now, these methods are intended for in-house use and will break without the supporting directory structure.
The actual directory_defs.json is ignored, but a sample json file is supplied in docs.

## Main function: fmri_analysis_load_funcs
At the beginning of a project, you will need to define a few structures in a directory_defs.json file and build a connectivity analysis of fMRI data in CONN. Once the definitions and connectivity matrices for each participant are available, you can call most any function in the modules and it will refer back to this load_funcs module. If groups are not defined or demographics are not available, it is likely that a module will break as these are designed to load off the bat and be available to multiple functions.


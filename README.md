# Exploring graph theory connectivity of neuroimaging data
Produces graph theory summaries based on connectivity matrices generated with the CONN toolbox. Given that the methods rely on matrices and dataframes, it is likely this package will be interchangeable with MEG or EEG data as well.

For now, these methods are intended for in-house use and will break without the supporting directory structure.
The actual directory_defs.json is ignored, but a sample json file is supplied in docs.

# Starting a new project
- Define the filepaths and fields in a directory_defs.json file
    - See docs/sample_directory_defs.json

- Build a connectivity analysis of fMRI data in CONN
    - Note which atlas you used for the CONN analysis. The same atlas will be referenced consistently for node information, including labels and approximate orthocenters (coordinates).
    - TODO: explain the required format/information in the atlas dataframe

- Draft a spreadsheet or dataframe with demographic, grouping, and correlative information. (This information is necessary for any of the group comparison functions to work


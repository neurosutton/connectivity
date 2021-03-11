"""Simple tool to make coordinates file from volume source."""

from mne import read_labels_from_annot, vertex_to_mni
import csv
import os.path as op

mri_root_dir = '/Users/joshbear/clinical/meg/anat'
scripts_dir = '/Users/joshbear/research/ied_network/scripts'

# parc = 'aparc.a2009s'
parc = 'HCPMMP1'
coords = []
hemi = {'lh': 0, 'rh': 1}

labels_parc = read_labels_from_annot('fsaverage', parc=parc,
                                     subjects_dir=mri_root_dir)

for label in labels_parc:
    ctr = label.center_of_mass()
    coords.append(vertex_to_mni(ctr, hemi[label.hemi], 'fsaverage')[0])

with open(op.join(scripts_dir, parc + '-coords.txt'), 'w+') as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=' ')
    csvWriter.writerows(coords)
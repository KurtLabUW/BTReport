# Midline Shift

Contains scripts and models for 3D midline shift quantification from CT/MRI data:
- Uses SynthMorph to align a healthy MNI Atlas to a subject's scan
- Uses transform from Synthmorph to register midline segmentation from MNI152 onto subject
- Outputs quantitative indicators including midline shift at each slice (in mm) and binary thresholds (>5 mm)


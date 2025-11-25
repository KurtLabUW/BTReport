# Midline Shift

Contains scripts and models for 3D midline shift quantification from MRI data:
- Uses SynthMorph to align a healthy MNI Atlas to a subject's scan
- Uses transform from Synthmorph to register midline segmentation from MNI152 onto subject
- Outputs quantitative indicators including midline shift at each slice (in mm) and binary thresholds (>5 mm)



## Setting up Synthmorph

SynthMorph is a great tool for registering arbitrary brain scans to each other, without needing specific preprocessing. SynthMorph is incredibly robust and can do cross-modality (MRI-to-CT) alignment as well. In this project, we use SynthMorph to register T1 scans to the MNI152 Atlas and vice-versa.

### 1. Installation

**Option 1** (takes ~1 hour)  
Install SynthMorph following the official [instructions](https://hub.docker.com/r/freesurfer/synthmorph). This will pull the image from DockerHub.

**Option 2** (recommended, takes ~20 minutes)  
Download `synthmorph_4.sif` directly from the provided Google Drive link.  Make sure `synthmorph_4.sif` is in a directory with sufficient space (the image is ~7 GB).


### 2. Set the $WRAPPER environment variable
Point to your installed synthmorph.sif with by setting:

```shell-session
export WRAPPER= path/to/synthmorph_4.sif
```
or inside Python with:
```python
os.environ['WRAPPER'] = 'path/to/synthmorph_4.sif'
```

### 4. Set the working directory of the SynthMorph container
`SUBJECTS_DIR` sets the working directory of SynthMorph, and you can specify paths relative to it.  If unset, `SUBJECTS_DIR` defaults to your current directory.

```shell-session
export SUBJECTS_DIR=/path/above/your/data
```
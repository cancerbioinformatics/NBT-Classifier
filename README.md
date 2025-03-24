# **A CNN-based _NBT-Classifier_ facilitates analysing different normal breast tissue compartments on whole slide images**

[Paper]() | [Cite]()

**Abstract:** Whole slide images (WSIs) are digitized tissue slides increasingly adopted in clinical practice and serve as promising resources for histopathological research through advanced computational methods. Recognizing tissue compartments and identifying regions of interest (ROIs) are fundamental steps in WSI analysis. In contrast to breast cancer, tools for high-throughput analysis of WSIs derived from normal breast tissue (NBT) are limited, despite NBT being an emerging area of research for early detection. We collected 70 WSIs from multiple NBT resources and cohorts, along with pathologist-guided manual annotations, to develop a robust convolutional neural network (CNN)-based classification model, named _NBT-Classifier_, which categorizes three major tissue compartments: epithelium, stroma, and adipocytes. The two versions of _NBT-Classifier_, processing 512 x 512- and 1024 x 1024-pixel input patches, achieved accuracies of 0.965 and 0.977 across three external datasets, respectively. Two explainable AI visualization techniques confirmed the histopathological relevance of the high-attention patterns associated with predicting specific tissue classes. Additionally, we integrated a WSI pre-processing pipeline to localize lobules and peri-lobular regions in NBT, the output from which is also compatible with interactive visualization and built-in image analysis on the QuPath platform. The _NBT-Classifier_ and the accompanying pipeline will significantly reduce manual effort and enhance reproducibility for conducting advanced computational pathology (CPath) studies on large-scale NBT datasets.

<p align="center">
    <img src="data/NBT.png" width="100%">
</p>

## Installation
To get started, clone the repository, install [HistoQC](https://github.com/choosehappy/HistoQC.git) and other required dependencies. 
```
git clone https://github.com/SiyuanChen726/NBT-Classifier.git
cd NBT-Classifier
conda env create -f environment.yml
conda activate tfgpu-env
```

## Implementation

WSI data is expected to be organised as follows:
```
prj_BreastAgeNet/
├── CLINIC/clinicData_all.csv
├── WSIs
│   ├── KHP/slide1.ndpi, slide2.ndpi ...
│   ├── NKI/slide1.mrxs, ...
│   ├── BCI/slide1.ndpi, ...
│   ├── EPFL/slide1.vsi, ...
│   └── SGK/slide1.svs, ...
```

First, implement HistoQC to detect foreground tissue regions:
```
python -m histoqc -c v2.1 -n 3 "/path/to/slides/*.ndpi" -o "/path/to/output/folder"
```
This step yields:
```
prj_BreastAgeNet/
├── WSIs
├── QC/KHP
│   ├── slide1/slide1_maskuse.png
│   └── ...
```

Then, use the following script to classify NBT tissue components:
```
python main.py \
  --wsi_folder /path/to/WSIs/directory \
  --mask_folder /path/to/QC/directory \
  --output_folder /path/to/Features/directory \
  --model_type TC_512 \
  --patch_size_microns 128
```

This step yields:
```
prj_BreastAgeNet/
├── WSIs
├── QC/KHP
│   ├── slide1/slide1_maskuse.png
│   └── ...
├── Features/KHP
│   ├── slide1/slide1_TC_512_probmask.npy     # This is the tissue classification results
│   ├── slide1/slide1_TC_512.png              # This visualises the tissue classification map
│   ├── slide1/slide1_TC_512_All.csv          # This saves all classified patches
│   ├── slide1/slide1_TC_512_cls.json         # This imports all classified patches into QuPath using the annotation_loader.groovy script
│   ├── slide1/slide1_TC_512_epi_(downsample,0,0,width,height)-mask.png      # This imports detected lobuels into QuPath using the mask2annotation.groovy script
│   ├── slide1/slide1_TC_512_patch.csv        # This saves the selected patches
│   ├── slide1/slide1_TC_512_ROIdetection.json     # This imports selected patches into QuPath using the annotation_loader.groovy script
│   ├── slide1/slide1_TC_512_bbx.png          # This visualises the selected ROIs
│   └── ...
```
 
 
For a full implementation of **_NBT-Classifier_**, please take a look at [notebook pipeline](pipeline.ipynb). 

# Notebook structure

Run the notebooks in this order:

1. `01EDA/01_EDA.ipynb` — inspect classes, image counts, dimensions, and corrupt files.
2. `02Preprocessing/02_preprocessing_split.ipynb` — create the reproducible stratified 70/15/15 manifest at `results/preprocessing/split_manifest.csv`.
3. `03Normalization/03_resnet50_normalization.ipynb` — resize RGB images to 224x224 and execute `tensorflow.keras.applications.resnet50.preprocess_input`.
4. `04DataAugmentation/04_data_augmentation_datasets.ipynb` — create separate `train_ds`, `validation_ds`, and `test_ds` datasets. Augmentation is enabled only for `train_ds`.

The notebooks expect the dataset at `PlantVillage/` relative to the repository root. The split manifest keeps training, validation, and test samples separate without duplicating the image files.

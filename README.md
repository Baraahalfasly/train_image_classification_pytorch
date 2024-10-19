# train_image_classification_pytorch

For training:
python train_resnet_on_custom_data.py --operation train

for testing:

python train_resnet_on_custom_data.py --operation test


The data should be structured as follows:

data--
    -train
          - glasses
          - watch
          - water_can
          - notebook
    -test
          - glasses
          - watch
          - water_can
          - notebook

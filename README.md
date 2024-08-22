# Hy-Tracker
Code for Hy-Tracker: A Novel Framework for Enhancing Efficiency and Accuracy of Object Tracking in Hyperspectral Videos
# HOT2023 Dataset
# Training
The training of Hy-Tracker consists of three parts:
1. Band Selection using Hierarchical Attention for Band Selection (HASBS):
   
   i. The HOT2023 dataset consists of three different types of data: vis, nir and rednir. Therefore, we develop three HASBS, one for each type of data.
   
   ii. Run the main file of the band15, band16, and band25 folders under the HASBS folder using the appropriate link to the datasets.
3. Sequence model using GRU:
   
   i. Prepare the dataset by running the create_dataset.py under GRU_Network using the appropriate link to the datasets.
   
   ii. Run the training.py under the GRU_Network for the sequence model
5. YOLO training:

   i. Create the dataset for YOLO training by running nir_data_processing, rednir_data_processing and vis_data_processing files under the data_processing folder. We used all the training sets plus the first frame of the validation datasets.

   ii. Download the yolo.pt file from this link: [yolo.pt](https://drive.google.com/file/d/1GfZpbcW_5GQP2WVtt2vFIx-SW8pckVJK/view) and put it in pretrained folder

   ii. Run the training.py file to train the Yolo model.


# Tracking
Run the tracking file using the appropriate link to the validation datasets.


# HOT2022 Dataset
# training
1. Band Selection: Run the main file under the HASBS/band_16 folder for band selection
2. GRU: Create the dataset by running the create_dataset.py file under GRU-Network, then train the model by running training.py file.
3. YOLO: create the dataset by running the hsi_data_processing file under the data_processing folder. We used all the training sets plus the first frame of the validation datasets. Download the yolo.pt file from this link: [yolo.pt](https://drive.google.com/file/d/1GfZpbcW_5GQP2WVtt2vFIx-SW8pckVJK/view) and put it in pretrained folder and then Run the training.py file to train the Yolo model.
# tracking
1. Run the tracking file using the appropriate link to the validation datasets.

# If this work is helpful to you, please cite it as:
```bibtex
@article{islam2024hy,
  title={Hy-Tracker: A Novel Framework for Enhancing Efficiency and Accuracy of Object Tracking in Hyperspectral Videos},
  author={Islam, Mohammad Aminul and Xing, Wangzhi and Zhou, Jun and Gao, Yongsheng and Paliwal, Kuldip K},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}



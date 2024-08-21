# Hy-Tracker
Code for Hy-Tracker: A Novel Framework for Enhancing Efficiency and Accuracy of Object Tracking in Hyperspectral Videos
# Training
The training of Hy-Tracker consists of three parts:
1. Band Selection using Hierarchical Attention for Band Selection (HASBS):
   
   i. The HOT2023 dataset consists of three different types of data: vis, nir and rednir. Therefore, we develop three HASBS, one for each type of data.
   
   ii. Run the main file of the band15, band16, and band25 folders under the HASBS folder using the appropriate link to the datasets.
3. Sequence model using GRU:
   
   i. Prepare the dataset by running the create_dataset.py under GRU_Network using the appropriate link to the datasets.
   
   ii. Run the training.py under the GRU_Network for the sequence model
5. YOLO training:
   


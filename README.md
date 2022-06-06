**Tuberculosis Diagnosis using PyTorch, Model: VGG19.**

Dataset was obtained from real cases, with the organization ImageCLEFmed Tuberculosis. a dataset containing chest CT scans of 403 (283 for train and 120 for test) TB patients is used. Since the labels are provided on lung-wise scale rather than CT-wise scale, the total number of cases is virtually increased twice. 

_Serge Kozlovski, Vitali Liauchuk, Yashin Dicente Cid, Aleh Tarasau, Vassili Kovalev, Henning Müller, Overview of ImageCLEFtuberculosis 2020 - Automatic CT-based Report Generation, CLEF working notes, CEUR, 2020._

The dataset has been modified into .csv file to avoid copyright infringement.

-------------------------------

**Task Description:**

In this task participants have to generate automatic lung-wise reports based on the CT image data.
Each report should include the probability scores (ranging from 0 to 1) for each of the three labels and for each of the lungs (resulting in 6 entries per CT). 

The resulting list of entries includes: LeftLungAffected, RightLungAffected, CavernsLeft, CavernsRight, PleurisyLeft, PleurisyRight.

-------------------------------

**Keywords:** 

· Automatic CT Report Generation 

· Deep Learning 

· Convolutional Neural Network 

· Tuberculosis Diagnosis 

· 3D Medical Image Analysis.

--------------------

**Implementation:**

The result shows the strength of our model trained in a small dataset with highly unbalanced label distributions, leading to 4th place on the Leaderboard of ImageCLEF2020 Tuberclosis Challenge, with mean AUC of 0.767 on the test dataset

_Dependencies:_


Python (Version 3.7 - 3.9)

PyTorch (PyTorch 3.x)

Pickle (Latest)

Numpy (Latest)

ArgParse (Latest)

json (Latest)

---------------------------------------

The python codes here, includes:

Loader File   : Loads the .csv files as input.

Model File    : VGG19 Model configuration.

Train File    : Training functions for loaded input data located in .csv.

Evaluate File : Summarize the results.

------


**About The Paper and Challenge:**

When referring to the ImageCLEFtuberculosis 2020 task or about our method, etc. please cite the following publication:

Mossa A.A., Eriş H., Çevik U., “Ensemble of Deep Learning Models for Automatic 
Tuberculosis Diagnosis Using Chest CT Scans: Contribution to the ImageCLEF-2020 
Challenges”, CLEF 2020, 22-25 September 2020, Thessaloniki, Greece.

_View full-paper:_

_http://ceur-ws.org/Vol-2696/paper_64.pdf_

**Abstract:** 

Tuberculosis (TB) is a bacterial infection that mainly affects the lungs. It is a potentially serious disease killing around 2 million people a year. Nevertheless, it can be cured if treated with the right antibiotics. However, manual diagnosing of TB can be difficult, and several tests are usually needed. Consequently, automated diagnosis of TB based on chest Computed Tomography (CT) images are of great interest. Recently, deep learning (DL) algorithms, and in particular convolutional neural network (CNN), due to the ability to learn low- and  high-level discriminative features directly from images in an end-to-end  architecture, have been shown to be the state-of-the-art in automatic medical image analysis. In this work, we developed a DL model for automated TB diagnosis using different CNN architectures with the ensemble method on 2D reconstructed images from 3D chest CT scans. The CNN based method proposed in this study includes Multi-View and Triplanar CNN architectures using pre-trained AlexNet, VGG11, VGG19 and GoogLeNet feature extraction layers as a backend. Using five-fold cross validation, the average AUC, Accuracy, Sensitivity and Specificity of the proposed ensemble method were 0.799, 77.1, 0.57 and 0.824, respectively, for multi-label binary classification on the ImageCLEFtuberculosis 2020 challenging training dataset of the lung-based automated CT report generation task, which is a well-benchmarked public dataset running every year since 2017. The result shows the strength of our model trained in a small dataset with highly unbalanced label distributions, leading to 4th place on the Leaderboard, with mean AUC of 0.767 on the test dataset. 
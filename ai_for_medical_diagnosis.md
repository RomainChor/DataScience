# AI for medical diagnosis
Based on [Deeplearning.ai course](https://www.coursera.org/learn/ai-for-medical-diagnosis/home/welcome). 

AI (Machine learning) have various applications in many fields of medicine. For instance: 
- Ophtalmology: https://www.nature.com/articles/s41433-018-0269-y
- Dermatology: https://www.nature.com/articles/nature21056
- Histopathology: https://pubmed.ncbi.nlm.nih.gov/30312179/

The available data can be of many forms: tabular, images, etc.  

Chest X-rays are one of the most common imaging procedures in medicine. Their interpretation is critical for the detection of pneumonia, lung cancer, etc.  
Very often, medical images such as **whole-slide image** are in very high-resolution (100.000x100.000 pixels). Therefore a single image must be broken into several patches before being fed to an algorithm. 


## Training models with medical data: challenges
There are 3 key challenges when using medical data for models training.

### Class imbalance
The **prevalence** (frequency) of disease in the world is low thus only a few "positive" examples can be found in datasets.  

Solutions to tackle class imbalance include **loss weighting** and **resampling**.

### Multi-task
Multiple diseases can be identified on a single image. Using a different model for each task is possible but costful and does not optimally exploit the extracted features from the image.   

Using a single model to identify multiple diseases at the same time requires a **multi-task (multi-label) loss** function

### Datasets size
Pre-trained neural networks (e.g. DenseNet, EfficientNet, Inception) often require millions of labelled images. In medicine, there are definitely not millions of labelled images available for training. Indeed, manually labeling medical data is time-consuming and requires experts.  

One solution is to use **data augmentation** to acquire new training examples. When using this technique, one may first ask these questions: 
- Do augmentations reflect real-world variations?
- Do augmentations preserve the label? e.g. *dextrocardia* is a condition characterized by the heart point to the right side of the chest. Doing vertical flipping would not preserve the label.

In addition, **transfer learning** is often used when training NN with medical data. 

## Testing models in medicine: challenges
### Patient overlap 
This happens when a patient is associated to several examples that are found in both training and test sets. This can lead to **over-optimistic** test set performance since the model can learn to recognize specific aspects of the patient and not necessary the ones we are interested in.

To solve that, make sure all examples related to a patient are in the same set. Patient overlap in medical data is a part of a more general problem in machine learning called **data leakage**.

### Sampling
One question is how should train/validation/test sets be sampled from a dataset. As said earlier, we often must face class imbalance. Therefore we usually want to first sample the **test set first** such that it contains at least $X\%$ of samples from the minority class. Afterwards we apply the same sampling strategy for the **validation** set. The **training** set then gets the remaining patients; it will probably have a small fraction of samples from the minority class thus one must still handle class imbalance.

### Ground truth/reference standard
Labeling images is done by experts: they set the **ground truth**. When experts disagree on the way to interpret an image, there is an **inter-observer disagreement**.   

One way to settle the ground truth is by **consensus voting** which means that either we choose the majority vote or we let experts discuss until a consensus is found.  
Alternatively, one can use an additional source of information (addition medical testing) if available.

## Models evaluation for classification
For classification, we use **accuracy** by default. In terms of conditional probabilities: 
$$ Accuracy = \mathbb P(1 | disease) \mathbb P(disease) + \mathbb P(0 | normal) $$
Which is equivalent to:
$$ Accuracy = Sensitivity. Prevalence + Specificity. (1 - Prevalence) $$  
with:
- Prevalence: disease cases rate, $\mathbb P(disease)$
- Sensitivity: true positive rate (TPR), $\mathbb P(1 | disease)$
- Specificity: true negative rate (TNR), $\mathbb P(0 | normal)$

Accuracy does not give a very trustworthy evaluation of a model's performance if there is class imbalance. One would prefer to use a **confusion matrix** and compute sensitivity, specificity, PPV ($\mathbb P(disease | 1)$) and NPV ($\mathbb P(normal | 0)$) from there.

A very wide-used metric for models evaluation is the **ROC curve** which allows us to visually plot the sensitivity of a model against the specificity of the model at different **decision thresholds**. Choosing decision thresholds is key in medicine and depends on the treated problem.  

Usually, evaluating a model is done on a subset of a population. Therefore, the obtained model's accuracy is not **exact**. It should be given with a **confidence interval** (CI).  

**Precision-Recall** is a useful measure of success of prediction when the classes are very imbalanced. 
-   Precision is a measure of result relevancy and that is equivalent to the previously defined PPV.
-   Recall is a measure of how many truly relevant results are returned and that is equivalent to our previously defined sensitivity measure.

The precision-recall curve (PRC) shows the trade-off between precision and recall for different thresholds. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate.


## Image segmentation in medicine
Medical image segmentation is not only being able to classify whether a medical image contains a disease or not, but also outline **which parts** of the image contained the disease.  
One very wide-used medical images for segmentation is MRI images. MRI images are not 2D images like X-Rays but instead are made of (2D) **sequences** of **layers/slices** of a 3D volume. Each sequence is associated to a **channel** hence an MRI image contains multiple 3D volumes.  

### Channels misalignment
To obtain an image, a slice/layer is picked then all the channels must be combined. One challenge with combining these sequences is that they may **not be aligned** with each other. 
For instance, if a patient moves between the acquiring of each of these sequences,
their head might be tilted in one sequence compared with the others.  
A preprocessing approach that is often used to fix this is called **image registration**.

### 2D-approach segmentation
One way to apply segmentation to MRI images is to break the 3D image into multiple 2D slices. Each slice is passed to a segmentation model (as a simple 2D image). Then the output slices (masks) are recombined to form a 3D volume of the segmentation.  
The major drawback of this approach is losing the 3D context of the MRI image.

### 3D-approach segmentation
To tackle this problem, an MRI image can be treated as a whole to obtain a **3D segmentation map**. Such volume is usually too large to be fed at one time to a segmentation model.  
Therefore, the 3D MRI volume is broken up into many 3D sub volumes. Each of these sub volumes has some width, height, and depth context.  

So like in the 2D approach, one can feed in the sub volumes one at a time into the model and then aggregate them at the end to form a segmentation map for the whole volume. Note that we might still lose important spatial context. For instance, if there is a tumor in one sub volume, there is likely to be a tumor in the sub volumes around it too.  

The silver lining with the 3D approach is that that we're capturing some context in all of the width, height, and depth mentions.

### U-Nets
One of the most popular architectures for segmentation has been the **[U-Net](https://arxiv.org/pdf/1505.04597.pdf)**. The U-Net was first designed for biomedical image segmentation and demonstrated great results on the task of cell tracking. It can achieve relatively good results, even with hundreds of examples.  
The U-Net consists of two paths: 
- **Contracting path**:  a typical convolutional network. The convolution operation here is called a down convolution. The key here, is that in the contracting path, the feature map gets spatially smaller, which is why it's called a contraction.
- **Expanding path**: in some ways is doing the opposite of the contracting path. It's taking the small feature maps through a series of up-sampling and up-convolution steps to get back to the original size of the image. It also concatenates the up-sample representations at each step with the corresponding feature maps at the contraction pathway! At the last step, the architecture outputs the probability of tumor for each pixel in the image.

In the case of 3D segmentation, the 3D U-Net works similarly with replacing 2D convolution operations by 3D convolutions. 

### Loss functions for 3D segmentation
The loss function used to compare the output segmentation map $P$ with the ground truth map $G$ has to consider every pixel of the image and all volumes.  
One convenient choice is the **soft dice** loss. This loss works very well with imbalanced data. In brain MRI images for instance, tumorous tissues can occupy a very small region of an image.
$$ L(P, G) = 1 - \frac{2 \sum_{i=1}^n p_i g_i}{\sum_{i=1}^n p_i^2 + \sum_{i=1}^n g_i^2 }$$
where $p_i$ (resp. $g_i$) are the pixels of the map $P$ (resp. $G$)


## Limits of AI for medicine in the real-world
### About the data
To be able to measure the generalization of a model on a population that it hasn't 
seen, we want to be able to evaluate on a test set sampled from the new population. 
This is called **external validation**. External validation can be contrasted with **internal validation**, when the test set is drawn from the same distribution as the training set for the model.  
If we find that we're not generalizing to the new population, then we could get a few more samples from the new population to create a small training and validation set and then **fine-tune** the model on this new data.  

Moreover, most of the case studies of ML for medicine use **retrospective data** i.e. historically labeled data to train and test algorithms. However, to understand the utility of AI models in the real-world, they need to be applied to real-world data or prospective data.   
With retrospective data, there are often steps taken to process and clean the data, but in the real-world, the model has to work with the raw data.

### About models 
Another challenge for the real-world deployment of AI models is that we need metrics to reflect **clinical application**. In the real world, we want to be able to look at the effect of our model on real patients. Specifically, we care about measuring whether the model eventually improves **patient health outcomes**.  

One approach to this challenge includes **decision curve analysis**, which can help quantify the net benefit of using a model to guide patient care.  
Another approach is to see what happens in the setting of a **randomized control trial** where we compare patient outcomes for patients on whom the AI algorithm is applied versus those on whom it is not applied.  

Also, one wants to build algorithms without blind spots or unintended biases.  

Finally one of the major challenges and opportunities to applying AI medical models in the real world is achieving a better understanding of how these algorithms will interact with the decision-making of clinicians. In other words, **models interpretation** is key to apply AI to medicine.

# Learning Utsav Challenge - 30-Day Self Learning Journey

### Dilli Hang Rai  
📅 Starts from: Oct 3, 2024  
🔗 **Twitter**:https://x.com/dilli_hangrae

---

## 🚀 Introduction

Welcome to my **30-Day Self Learning Challenge** as part of the **Learning Utsav Challenge**, starting on Ghatasthapana and ending on Bhaitika! This initiative is designed to turn the Dashain-Tihar festive season into a time for learning and growth. Sponsored by **Evolve IT Hub Nepal** and **Programiz**, and organized by the **KEC Electronics Club** with support from student clubs across Nepal, this challenge encourages self-learning in various technical domains.

### 📚 What I'm Learning:
I'll be focusing on **AI**, **Machine Learning**, and **Deep Learning** throughout the next 30 days, sharing my progress and key learnings daily on **Twitter** using the hashtag #LearningUtsav. You can follow my journey and see the progress in this repository as well.

---

## 🏆 Challenge Overview

- **Challenge Duration**: Oct 3, 2024 (Ghatasthapana) – Nov 2, 2024 (Bhaitika)
- **Platforms**: Twitter (#LearningUtsav)  
- **Focus Areas**: AI, Machine Learning, Deep Learning

---

## 📝 Daily Progress

| Day | Topic & Twitter Post 
| --- | ------------------------------------------------------------------------------------------
| 1   | Introduction to Machine Learning, https://x.com/dilli_hangrae/status/1841763188117012784
| 2   | Exploring concepts on parameters,models,vectors,data points of objects, https://x.com/dilli_hangrae/status/1842117445768839259
| 3   | Intro to Probability & Statistics, Distributions, Discrete and Continuous, PMF,PDF, Bernoulli Distribution , https://x.com/dilli_hangrae/status/1842409414663213375
| 4   | Joint Probability Distribution, PDF, Multi-variate Gaussian Distribution,Marginal Probability,Conditional Probability,AI Perspectives: Thinking and acting rationality and humanly, https://x.com/dilli_hangrae/status/1842784087825899934
| 5   | Expectations with their properties, independence, linearity,co-variance, variance,Bernoulli Cov,emprical mean and co-variance, https://x.com/dilli_hangrae/status/1843173089397612622
| 6   | Linear Algebra,Vector vs Scalars, Vector Space, Matrices, Vector Product, Span and Basis of Vector, Matrices as Linear Transformations, Representation of Graph, Dimensionality reduction using matrices,Vector Scaling, lINEARLY DEPENDENT AND independent, 2D & 3D Vectors Space, Polar and Catesian Co-ordinates, Outer Product, Inner Product, Dot Product and Hadamard Product, https://x.com/dilli_hangrae/status/1843523500608503902
| 7   | Dot Product in ANN,Eigen Vectors and Values, Eigen Decomposition, Matrix as Linear Transformation, Roles of Eigen with animation plots , https://x.com/dilli_hangrae/status/1843882103689359408
| 8   | Quick Theoretical Tour on Machine Learning - Linear and Logistic Regression with some Mathematical notations, Loss function, training and testing dataset, Knowledge representation in AI, issues in KB repsentation and methods to represent the Knowledge, Maximum Likelihood Estimation, Parameteric vs Non-Parameteric Model of Function, https://x.com/dilli_hangrae/status/1844293749322752346
|  9  | Cross-Validatios, Regularization L1 and L2,K-NN, K-Means Clustering, SVM (non-linear), AI agents, PEAS, Sensor-Actuators-Effectors, Bias-Variance Trade-off, https://x.com/dilli_hangrae/status/1844601197946437918
|  10 | Statistical Method: Finding beta0 and beta1 y = beta0 + beta1* X where X is input and predicting the outcome 0 or 1 using Statistics of logistic regression , https://x.com/dilli_hangrae/status/1844945255583580320
|  11 | Revised: Vectors, Matrices, Basis, Dependencies, Eigenvalues/Vectors, Decomposition, Statistical/Probability Concepts (PDF, PMF, Marginal Probability, Independence, Variance/Covariance, Distributions), Chain Rule, Expectation, Worked with various static datasets in Logistic Regression, https://x.com/dilli_hangrae/status/1845352394428908024
| 12  | Supervised Learning: Classification vs Regression: Linear Regression vs Sparse Regression (using less features) and Logistic Regression vs Sparse Logistic Regression and Non-Linearity Model Tree Based: Random vs XG Boosted Random Forest(building the tree sequentially each tree step by step and correct the previous error tree with the current tree) with visualizations plots, https://x.com/dilli_hangrae/status/1845706594757353833
| 13  | Z-score normalizations, transformation, differences in scales and using sparse, featurization, filling the missing values for both classification and regression, knowledge vs facts vs information vs representations, propositional as a knowledge representation method with operators and backward+forward knowledge representations methods, https://x.com/dilli_hangrae/status/1846050651559940229
|  14 | Detailed Study on Overfitting, Underfitting, Prevention of Overfitting: Cross-Validation and Test/Split Dataset, Bias-Variance TradeOff, Logistic Regression Loss Function Derivation, https://x.com/dilli_hangrae/status/1846445899838595257
|  15 |Getting into CNN,parameters, kernel operator in CNN,Motivation of Convolutional Neural Networks,Edge detection using gaussian filter and noise modeled via Gaussian Probability Distribution, https://x.com/dilli_hangrae/status/1846853957127491867
|  16 | Continuing CNNs, Pooling, Max vs Average Pooling, Applying ReLu and filters in the entire image and viewing the EKG Signals filters plots, Propositional Logic vs Predicate Logic vs First-Order Predicate Logic used in Knowledge Representation, CNNs in action and reading the block steps of FCN vs CCN, channels, batch size, input-output features and size, Receptive Field, 
|| Coding parts are:
|| Load EKG Signal, Read 1000 samples of EKG data from the MIT-BIH Arrhythmia Database (record '100'),
|| Extract Signal and Time Axis: Extract the EKG signal and create a time array in seconds,
|| Apply Gaussian Smoothing: Use a Gaussian filter to reduce noise in the signal,
|| Edge Detection (Gradient): Calculate the gradient (first derivative) of the smoothed signal to detect sharp transitions (peaks),
|| Noise Reduction: Compute noise reduction by subtracting the smoothed signal from the original,
|| Visualization: Plot the original signal, smoothed signal, edge detection result, and noise reduction in a 4-plot grid,
||Sobel Edge Detection: Apply a Sobel filter to highlight sharp changes in the signal,
||Final Plot: Plot the original signal and Sobel-filtered edge detection results side by side, https://x.com/dilli_hangrae/status/1847138828794187780
| 17  | CNNs Continue, Max Pooling as Translation invariance, Counting the number of parameters, understanding the CNN Architectures and Network Diagrams,Strides, Implementation of the simple CNN model, https://x.com/dilli_hangrae/status/1847550062844662121
|| Steps of CNN Operation
|| Load the MNIST Dataset: The dataset is split into training and testing sets.
|| Data Preprocessing:
|| Normalization: Pixel values are scaled to the range [0, 1].
|| Reshaping: Each image is reshaped to include a channel dimension.
|| One-Hot Encoding of Labels: Convert class labels to binary class matrices.
|| Model Creation: A Sequential model is constructed with multiple layers:
|| Convolutional Layers: Extract features from the images.
|| Max Pooling Layers: Reduce the dimensionality while retaining important features.
|| Flatten Layer: Convert the 2D feature maps into a 1D vector.
|| Dense Layers: Perform classification on the flattened vector.
|| Dropout Layer: Regularize the model by randomly setting a fraction of input units to 0 during training.
|| Model Compilation: The model is compiled with an optimizer and loss function.
|| Model Training: The model is trained on the training data with early stopping.
|| Model Evaluation: The model is evaluated on the test data to assess performance.
|| Plotting Results: Visualize training and validation accuracy and loss over epochs.
| 18  | Theoretical Knowledge on CNNs-Face Detection, Classification vs Regression in CNNs, Handling the Input image, Siamese Architecture, Transfer Learning, Idea on how facebook is handling the large billion of Face Detection/Classifier, Knowledge Representation on Propositional Logic(PL)-- tautology, validity, well-formed-formula, Inference using Resolution, Backward Chaining and Forward Chaining, Code Explanation - https://github.com/Dilli822/Learning-Utsav-AI_ML-/blob/main/30Days%20Self%20Learning%20Challenge/Day18/day18.md/, tweet - https://x.com/dilli_hangrae/status/1847945533417247112
| 19  | CNN built in model SSD MobileNet model is to detect objects within the images (Object Detection), Data augmentation, use of data-augmentation in CNN, FOPL to PL, CODE EXPLAIN: https://github.com/Dilli822/Learning-Utsav-AI_ML-/blob/main/30Days%20Self%20Learning%20Challenge/Day19/day19.md, TWEET: https://x.com/dilli_hangrae/status/1848257999183471049
| 20  | Glimpse into CNN architecture: MobileNet, GoogleNet,ResNet, Custom CNN Model vs MobileNet, GoogleNet,ResNet Models,Knowledge Repersentation Unification and Lifting, https://x.com/dilli_hangrae/status/1848659400300396602
| 21  | Logistic Regression Cost vs Loss Function with Maths,plots and animations, decision boundary, https://x.com/dilli_hangrae/status/1849058116538978481
| 22  | Logistic Regression --> Maximum loglikelihodd, learning pi(x), gardient descent, derivative of sigmoid function, visualizations of gradient descent, GOOGLE COLAB ON SOME PROBABILITY CONCEPTS , https://x.com/dilli_hangrae/status/1849349940281331984
| 23  | Deep Learning Foundations Continuing: Sigmoid,Chain Rule, Model Fitness- R-CNN Kaggle Object Segmentation, https://x.com/dilli_hangrae/status/1849682504925708479
| 24  | Deep Learning Foundations Continuing:Chain Rule in LR, importance of feature selection, importance of structured data inputs, lasso regularization and hitting with penalties, mxing and stacking the layers in LR, applying lambda or adjusting lambda with regularization terms, https://x.com/dilli_hangrae/status/1850085169182765387
| 25  | Deep Learning Foundations Continuing: Chain Rule everywhere, backpropagation,activation functions, optimizers, animations and plots, theories on Deep Learning non-linearity and KB CNF, https://x.com/dilli_hangrae/status/1850416977128681923
| 26  | Deep Learning Foundations Continuing: Chain Rule everywhere, backpropagation,modularity, organized flow graph of backprogatation, visualization and animation of feature selections, concept of locality, cost function regularized in both linear and logistic regression maths, https://x.com/dilli_hangrae/status/1850768449922056645
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |

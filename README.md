# Applied-Machine-Learning-
COMP551 - Prof. Joelle Pineau - McGill University

From https://github.com/Pulkit-Khandelwal/COMP551-Applied-Machine-Learning

=======================================================

Course Skeleton 

Lec1: Introduction

Lec2: Linear regression

Lec3: Linear regression

Lec4: Linear classification

Lec5: Linear classification

Lec6: Performance analysis and error estimation

Lec7: Decision trees

Lec8: Instance-based learning

Lec9: Feature construction and selection

Lec10: Ensemble methods

Lec11: Support vector machines

Lec12: Support vector machines

Lec13: Unsupervised learning

Lec14: Neural networks

Lec15: Neural networks (cont'd)

Lec16: Deep learning

Lec17: Deep learning (cont'd)

Lec18: Semi-supervised learning / Generative Models

Lec19: Bayesian Inference

Lec20: Gaussian Processes

Lec21: Bayesian Optimization

Lec23: Parallelization for large-scale ML

Lec24: Missing data

=======================================================

Course Content 

Lec1: Introduction

    Basic Algebra and Probability

    • http://www.cs.mcgill.ca/~dprecup/courses/ML/Materials/prob-review.pdf
    • http://www.cs.mcgill.ca/~dprecup/courses/ML/Materials/linalg-review.pdf

Lec2: Linear regression

    Least Square Solutions and Gradient Descent method

Lec3: Linear regression

    Regressio Expansion; Overfitting and Cross-validation

Lec4: Linear classification

    Discrimnative Learning (Logistic Regression) 

Lec5: Linear classification

    Generation Learning (LDA: Same Cov & QDA: Diff Cov)

    Naive Bayes - Text classification example - Laplace smoothing(Unobserved word)

    Gaussian Naive Bayes: Diagonal Cov

Lec6: Performance analysis and error estimation

    Sensitivity&Specificity - Accurary - Precision&Recall 

    K-CV - ROC - AUC

    Baseline Comparsion of Classification and Regression Problems

Lec7: Decision trees

    Recursive Learning 

    Entropy - Conditional Entropy - Information Gain

    Pros & Cons

Lec8: Instance-based learning

    Parametric Supervised Learning v.s. Instance-based Learning 

    Distance metric scalering - Domain Specific Weighting Functions 

    Pros & Cons and Application of Lazy Learning 

    Project2

Lec9: Feature construction and selection (NLP)

    Natural Language toolkit: http://www.nltk.org/
    
    Words (Binary - Absolute frequency - Relative frequency -Stopwords - Lemmatization - TF*IDF - N-grams)
    
    Word2Vec (Bag-of-words; Skip-gram) 

    (Wrapper methods: PCA) (Embedded methods: Lasso & Ridge) (Variable Ranking: Sorcing Functions) 
    
Lec10: Ensemble methods

    Ensemble the results of learners with slightly 
    
    different datasets(Bagging[Var-]) or 
    
    different training procedure(Random Forests[Var-], Extremely randomized Trees[Bias-])
    
    Boosting[Bias-] (More weight on weak learners) - AdaBoost (auto-adapt error rate) 

Lec11: Support vector machines

    Linear Classifier - Perceptron - Dual Representation
    
    Lagrange multiplers transformation

Lec12: Support vector machines2(Non-linear)

    1, Relax the constraints
    
    [image](https://user-images.githubusercontent.com/77312114/116800792-cb295780-ab36-11eb-8568-67e6dd6adc76.png)

    2, Feature mapping
    
    Using Kernel (Dot product) 
    [image](https://user-images.githubusercontent.com/77312114/116800870-6c181280-ab37-11eb-8e10-afde36b69bd5.png)
    
    Kernel types (Normal kernel - Radial basis/Gaussian kernel - Sigmoiddal kernel - 
    Diffusion kernel(graphs) - String kernel(protein)) 

Lec13: Unsupervised learning

    K-means (Fixed Variance) - Gaussian Mixture Model (Expectation Maximization Modeling) 

    Anomaly Detection (Use Generative approach) - Dim reduction - Autoregression - Autoencoding
    
Lec14: Neural networks

    Notations (N, H, N-H-1)
    
    Organizing the trianing data (Stochasi]tic GD, Batch GD, Mini-batch GD)

Lec15: Neural networks (cont'd)

    Relu
    
    Encoding methods (one-hot, 1-of-k, 1-to-n, thermometer)

Lec16: Deep learning

    Autoencoder - Stacked autoencoders
    
    Regularization(sparse): Weight tying, Penalize the hidden units output or the average output 
    
    Denoising(robust): Additive Gaussian noise, Dropout, Batch-norm
    
    Training protocols: Purely supervised(speech etc.) Supervised-classider-on-top(fewer labeled) 
    [image](https://user-images.githubusercontent.com/77312114/117086492-3c5b4b80-ad7f-11eb-8634-f01731632acb.png)
    
    CNN:
    www.image-net.com 
    
    Ideas: Local receptive fields, Shared parameters matrix, Pooling between neighbourood
    
    Layers: Convolutional - Pooling - Fully connected 
    

Lec17: Deep learning (cont'd)

Lec18: Semi-supervised learning / Generative Models

Lec19: Bayesian Inference

Lec20: Gaussian Processes

Lec21: Bayesian Optimization

Lec23: Parallelization for large-scale ML

Lec24: Missing data



















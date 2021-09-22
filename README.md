=======================================================
=======================================================
# Hands-On Machine Learning with scikit-learn, keras and Tensorflow (2 edtion) (Example Done, Exercise doing)

https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

(website with code example, using tensorflow2)
https://github.com/ageron/handson-ml2


### CH16 NLP with RNNs and Attention

>>> 16.1 Character RNN 

16.1.1 Get file

    filepath = keras.utils.get_file('txt', url) 

16.1.2 Tokenize 

    tokenizer = keras.preprocessing.text.Tokenizer(char_level = True) 

16.1.3 Encode all the text 

16.1.4 Chopping into Windows

    dataset = dataset.window(window_length, shift=1, drop_remainder=True)

16.1.5 Flatten, batch, and shuffle the data

16.1.6 Train the model//

16.1.7 Stateful RNN

16.1.8 Use batches containing a single wiindow 

16.1.9 In the first layer, set 'stateful = True'
    
>>> 16.2 Sentiment Analysis 

16.2.1 using the IMDb reviews 
    
    #Load the preprocess the data 
    
    def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300) 
    X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>",b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']",b" ")
    X_batch = tf.strings.split(X_batch) 
    return X_batch.to_tensor(default_value = b"<pad>"), y_batch  
    
16.2.2 Build the vocabulary, keep only 10,000 words

    from collections import Counter 

16.2.3 Masking, set 'mask_zero = True'

16.2.4 Resuing Pretrained Embeddings 

    import tensorflow_hub as hub 
    hub.KerasLayer(url, dtype=tf.string, input_shape=[], output_shape=[50]),

>>> 16.3 Encoder - Decoder 

16.3.1 Bucketing 

    tf.data.experimental.bucket_by_sequence_length() 

16.3.2 Addons project 

    import tensorflow_addons as tfa

16.3.3 Bidirectional RNNS 

    keras.layers.Bidirectional(keras.layers.GRU(10, return_sequences = True) 

16.3.4 Beam Search (keeping k candidates in each step) 
    
    # wraps all decoder
    decoder = tfa.seq2seq.beam_search_decoder.BeamSearchDecoder()
    # copy last state of encoder
    decoder_initial_state = tfa.seq2seq.beam_search_decoder.tile_batch()
    # pass to decoder
    outputs, _, _ = decoder()

>>> 16.4 Attention

16.4.1 Luong Dot Product attention 

    tfa.seq2seq.attention_wrapper.LuongAttention()

16.4.2 Position Encoding

16.4.3 Multi-head attention

=======================================================
=======================================================
# An introduction to Statitical Learning with Application in R (1 edition) (Exercise done with R)

1, The Book

http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf

CH1 Introduction

CH2 Statistical Learning (Model Accuracy)

CH3 Linear Regression

CH4 Classification (Logistic Regression, Linear Discriminant Analysis, Quadratic Discriminant Analysis, K-Nearest Neighbors)

CH5 Resampling Methods (Cross Validation, The Bootstrap)

CH6 Linear Model Selection and Regularization (Subset-Stepwise, Ridge-Lasso, Principal Components Regression-Partial Least Squares)

CH7 Moving Beyond Linearity (Ploynomial Regression, Splines, Generalized Additive Models)

CH8 Tree-Based Methods (Decision Trees, Bagging, Random Forests, Boosting)

CH9 Support Vector Machines (Maximal Margin Classifer, SVMs)

CH10 Unsupervised Learning (PCA, K-Means Clustering, Hierarchical Clustering)

2, The Course Video

https://lagunita.stanford.edu/courses

3, The exercise in R

https://blog.princehonest.com/stat-learning/


=======================================================
=======================================================
# Applied-Machine-Learning- (Lecture Done)

COMP551 - Prof. Joelle Pineau 

From https://github.com/Pulkit-Khandelwal/COMP551-Applied-Machine-Learning

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

Course Notes

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
    

Lec17: Deep learning (RNN)

    Arbitrary topology - Directed cycle(delay) 
    
    Backprop(tied weight on the unrolled RNN) 
    
    LSTMs & Bi-LSTMs
    
    Neural Language Modeling (Continuous space word representation - Nonlinear hidden layer - Softmax normalization) 

    Evaluation(Perplexity): Machine translation (BLEU, METEOR) Text summarization (ROUGE)

Lec18: Semi-supervised learning / Generative Models

    Boundary shift
    
    Self-training algorithm (warpper method, mistakes reinforce) 
    
    Generative approach (Gaussian distribution boundary shift - EM method) 
    
    Generative Adversarial Nets

Lec19: Bayesian Inference (estimate the uncertainty)

    Conjugate priors( Prior and Posterior have same family) 
    
    Examples: Tossing coins, 1-d Gaussian mean 
    
    Bayesian linear regression 

Lec20: Gaussian Processes (quantify uncertainty in regression) 

    Determining uncertainty (1, Determine posterior 2, Combine predictions) 
    
    Bayesian linear regression (Inference, Prediction) 
    
    Beyond linear regression (Grid of radial basis functions, Polynomial expansion) 
    
    Avoid using features (Instance-based learning, SVM-kernels) 
    
    Kernelizing Bayesian linear regression --> Gaussian Process (For small datasets) 

Lec21: Bayesian Optimization

    Choosing between linear methods
   [image](https://user-images.githubusercontent.com/77312114/117094712-1641a580-ad97-11eb-9079-9a4bdaaebb34.png)
   
    Hyperparameter optimisation for GPR & BLR 
   [image](https://user-images.githubusercontent.com/77312114/117095136-20b06f00-ad98-11eb-8a45-10c942842996.png)

        1,Marginal likelihood (used to choose features or kernels) 
        
        2,Automatic relevance determination(ARD) (High length scale - Low relevance) 
                
    Optimate unknown functions - Acquisition function (sample the highest value and uncertainty) 
    
    Application: hyperparameter-optimization

Lec23: Parallelization for large-scale ML

    Hadoop (Distributed File System - MapReduce - Advanced components) 
    
    Word count example (Parallel Execution) 
   [image](https://user-images.githubusercontent.com/77312114/117107165-b9081d00-adb3-11eb-8704-b6f82e32ae20.png)

    Parallel learning (Algorithm, Parameter search, Decision trees) 

Lec24: Missing data

    Missing types: 
    
    Missing Completely at Random (MCAR)
    Missing at Random (MAR) - Observed features depend
    Not Missing at Random (NMAR)
    
    Strategies(pros & cons):
    
    1, Deletion 
    (Listwise - Pairwise)
    
    2, Substitution methods
    (mean/mode sub)
    (include binary indicator variable)
    (regression imputation)
    
    3, Model-based methods
    (generative methods - EM)
    (multiple imputation - repeat and vote) 
    
    
    

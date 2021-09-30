=======================================================
=======================================================
# Hands-On Machine Learning with scikit-learn, keras and Tensorflow (2 edtion) (Example Done, Exercise doing)

https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

(website with code example, using tensorflow2)
https://github.com/ageron/handson-ml2

### CH2. End-to-End Machine Learning Project

### CH3. Classification

### CH4. Training Models

### CH5. Support Vector Machines

### CH6. Decision Trees

### CH7. Ensemble Learning and Random Forests

### CH8. Dimensionality Reduction

### CH9. Unsupervised Learning Techniques

### CH10. Introduction to Artificial Neural Networks with Keras

### CH11. Training Deep Neural Networks

### CH12. Custom Models and Training with TensorFlow

### CH13. Loading and Preprocessing Data with TensorFlow

### CH14. Deep Computer Vision Using Convolutional Neural Networks

### CH15. Processing Sequences Using RNNs and CNNs

### CH16. Natural Language Processing with RNNs and Attention

======
### >>> CH2 Regression 

>>> 2.1 The big picture 

>>> 2.2 Get the data 

>>> 2.3 Visualize the data 

>>> 2.4 Prepare the data 

>>> 2.5 Selection and Training 

>>> 2.6 Fine-tuning

### >>> CH3 Classification 

>>> 3.1 Binary Classifier 

3.1 Data preparation 

     # split
     X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
     # shuffle 
     import numpy as np 
     shuffle_index  = np.random.permutation(60000) 
     X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

3.1 Performence Measurement

     # cross validation
     from sklearn.model_selection import cross_val_score
     cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = 'accuracy') 

3.1 The baseline! (Especially for skewed datasets) 

3.1 The Confusion Matrix 

     # get prediction 
     from sklearn.model_selection import cross_val_predict
     y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
     
     # get matrix 
     from sklearn.metrics import confusion_matrix
     
3.1 Precision and Recall & F1

     from sklearn.metrics import precision_score, recall_score
     precision_score(y_train_5, y_train_pred)
     recall_score(y_train_5, y_train_pred)
     
     # F1 
     from sklearn.metrics import f1_score

3.1 Decide on the threshold 

     # set 'decision_function' method
     y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
method="decision_function")

     # compute all threshold for the curve
     from sklearn.metrics import precision_recall_curve 
     precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
     
     # plot the curve
     def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
         plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
         plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
         plt.xlabel("Threshold", fontsize=16)
         plt.legend(loc="upper left", fontsize=16)
         plt.ylim([0, 1])

     plt.figure(figsize=(8, 4))
     plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
     plt.xlim([-70000, 70000])
     save_fig("precision_recall_vs_threshold_plot")
     plt.show()

3.1 The ROC curve 

     # roc_curve class 
     from sklearn.metrics import roc_curve 
     fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
     
     # plot using matplot 

     def plot_roc_curve(fpr, tpr, label=None):
         plt.plot(fpr, tpr, linewidth=2, label=label)
         plt.plot([0, 1], [0, 1], 'k--')
         plt.axis([0, 1, 0, 1])
         plt.xlabel('False Positive Rate')
         plt.ylabel('True Positive Rate')
     plot_roc_curve(fpr, tpr)
     plt.show()
     
     # auc areas
     from sklearn.metrics import roc_auc_score 
     roc_auc_score(y_train_5, y_scores)
     
>>> 3.2 Multiclass Classifier 

3.2 Most algorithm handle it automatically 

### >>> CH4 Training Models

>>> 4.1 Linear Regression

4.1 Normal eqution

     # col 1
     X_b = np.c_[np.ones((100, 1)), X]
     # use numpy
     theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

4.1 LinearRegression()

    from sklearn.linear_model import LinearRegression
    # class 
    lin_reg = LinearRegression()
    # fit 
    lin_reg.fit(X, y)
    # param
    lin_reg.intercept_, lin_reg.coef_
    # prediction
    X_new = np.array([[0], [2]])
    lin_reg.predict(X_new)

>>> 4.2 Gradient Descent

4.2 Batch Gradient Descent

4.2 Stochastic Gradient Descent
    
    # SCDRegressor
    from sklearn.linear_model import SGDRegressor
    # 50 epochs, util loss = 1e-3, no regularization, learning_rate start at 0.1
    sgd_reg = SGDRegressor(max_iter=50, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
    
4.2 Polynomial Regression 

     # Add polynom to the feature 
     from sklearn.preprocessing import PolynomialFeatures 
     # It would try all the combinations of polynom terms
     poly_features = PolynomialFeatures(degree = 2, include_bias = False) 
     # Polynom term
     X_poly = poly_features.fit_transform(X) 
     # Feed to the linear reg now 
     lin_reg = LinearRegression() 
     lin_reg.fit(X_poly, y) 
     
4.2 Plot the learning curve 

     from sklearn.metrics import mean_squared_error 
     from sklearn.model_selection import train_test_split 
     from matplotlib import pyplot as plt
     
     # plot_learning_curces functions
     def plot_learning_curves(model, X, y):
         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
         train_errors, val_errors = [], []
         for m in range(1, len(X_train)):
             model.fit(X_train[:m], y_train[:m])
             y_train_predict = model.predict(X_train[:m])
             y_val_predict = model.predict(X_val)
             train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
             val_errors.append(mean_squared_error(y_val, y_val_predict))

         plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
         plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
         plt.legend(loc="upper right", fontsize=14)   # not shown in the book
         plt.xlabel("Training set size", fontsize=14) # not shown
         plt.ylabel("RMSE", fontsize=14)              # not shown
         
    # plot 
    lin_reg = LinearRegression() 
    plot_learning_curves(lin_reg, X, y) 
         

>>> 4.3 Regularized Linear Models

4.3 Ridge 

     # increase alpha leads to flatten the model 
     from sklearn.linear_model import Ridge 
     ridge_reg = Ridge(alpha = 1, solver = 'cholesky') 
     ridge_reg.fit(X,y) 
     
     # could also use SGD 
     sgd_reg = SGDRegressor(penalty = 'l2') 

4.3 Lasso 

     # Lasso 
     from sklearn.linear_model import Lasso 
     lasso_reg = Lasso(alpha = 1) 

4.3 A mixed method
     
     # ElasticNet 
     from sklearn.linear_model import ElasticNet
     elastic_net = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
     
4.3 Scale and plot 

     # jupyter CH4
     
4.3 Early stopping 

     # Early stopping with preset minimum_squared_error() 
     from sklearn.base import clone
     sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None,
                            learning_rate="constant", eta0=0.0005, random_state=42)

     minimum_val_error = float("inf")
     best_epoch = None
     best_model = None
     for epoch in range(1000):
         sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
         y_val_predict = sgd_reg.predict(X_val_poly_scaled)
         val_error = mean_squared_error(y_val, y_val_predict)
         if val_error < minimum_val_error:
             minimum_val_error = val_error
             best_epoch = epoch
             best_model = clone(sgd_reg)

>>> 4.4 Logistic Regression 

4.4 Normal 

     from sklearn.linear_model import LogisticRegression 
     log_reg = LogisticRegression() 

4.4 Softmax

     # multinomial logistic 
     # use 'C' to tune the L2 regularization param
     softmax_reg = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', C = 10)

### >>> CH5 Support Vector Machines

>>> 5.1 Linear SVM Classification

5.1 Linear SVM

    # important to scale the data
    # scale the data with a pipeline 
    from sklearn.pipeline import Pipeline 
    from sklearn.preprocessing import StandardScaler 
    from sklearn.svm import LinearSVC
    
    svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    # larger C, larger violation 
    ("linear_svc",LinearSVC(C=1, loss = "hinge")),
    ))
    svm_clf.fit(X, y)

>>> 5.2 Nonlinear SVM Classification

5.2.1 Nonlinear

    # add PolynomialFeatures() into the pipeline
    from sklearn.pipeline import Pipeline 
    from sklearn.preprocessing import PolynomialFeatures

    polynomial_svm_clf = Pipeline((
        ("poly_features", PolynomialFeatures(degree = 3)),
        ("scaler", StandardScaler()),
        ("svm_slf", LinearSVC(C = 10, loss = "hinge"))
    ))

5.2.2 SVC class and kernel tricks 

    # using SVC class
    # coef0 controls how much the model is influenced by high-degree polynomials versus low-degree polynomials
    from sklearn.svm import SVC 
    poly_kernel_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    
    # poly kernel
    ("svm_slf",SVC(kernel = "poly", degree = 3, coef0 = 1, C = 5))
    
    # Gaussian Radial Basis Function (RBF) kernel
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ))
 
 5.2.3 SVM Regression 
 
    from sklearn.svm import SVR 

    # epsilon = 'height of the street' / 2
    svm_poly_reg = SVR(kernel = "poly", degree = 1, C=100, epsilon=0.1)
    svm_poly_reg.fit(X,y)   

>>> 5.3 Under the hood 

### >>> CH6 Decision Trees 

>>> 6.1 Training 

6.1 DecisionTreeClassifier 

    from sklearn.tree import DecisionTreeClassifier

    # max_depth + 1 = decision boundaries
    tree_clf = DecisionTreeClassifier(max_depth = 2)

6.1 Visualization

    from sklearn.tree import export_graphviz

    export_graphviz(
            tree_clf,
            out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
            feature_names=iris.feature_names[2:],
            class_names=iris.target_names,
            rounded=True,
            filled=True
        )

6.1 CART

    # Classification and Regression Tree (CART)
    # Growing tree with less impurity 
    

>>> 6.2 Regularization 

6.2 Parameters

    # min_samples_split (the minimum number of samples a nodemust have before it can be split)
    # min_samples_leaf (the minimum number of samples a leaf node must have)
    # max_features (the maximum number of features that are evaluated for splitting at each node)
    # plot and see! 

>>> 6.3 Regression

6.3 DecisionTreeRegressor 

    # by mse
    from sklearn.tree import DecisionTreeRegressor 

    tree_reg = DecisionTreeRegressor(max_depth=2)
    tree_reg.fit(X,y)
    
    # def plot_regression_predictions()

### >>> CH7 Ensemble Learning and Random Forests

>>> 7.1 Bagging

7.1 Voting Classifiers
 
    # three classifiers 
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    # voting tool
    from sklearn.ensemble import VotingClassifier
    
    # four classifiers in total
    log_clf = LogisticRegression() 
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC() 
    voting_clf = VotingClassifier(estimators = [('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],)    
    
    # look at the accuracy_score 
    from sklearn.metrics import accuracy_score 
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))    
    
7.1 Bagging and Pasting

    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    # 500 trees, 100 instances for each one, using all cpu cores
    bag_clf = BaggingClassifier( 
        DecisionTreeClassifier(), n_estimators = 500,
        max_samples = 100, bootstrap = True, n_jobs = -1
    )    

7.1 Out-Of-Bag Evaluation

    oob_score = True

7.1 Random Patches and Random Subspace

    # Random Patches: sampling both instance and features
    # Random Subspaces: sampling features
    # BaggingClassifier class 
    # features param: max_features = , bootstrap_features = , 
    # instances param: max_sample = , bootstrap = ,

>>> 7.2 Random Forests

7.2 Feature inportance 

     rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
     rnd_clf.feature_importances_
    
>>> 7.3 Boosting

7.3 AdaBoost

    # squential training with instance weight updates 
    # the learning_rate controls the contribution of each tree 
    from sklearn.ensemble import AdaBoostClassifier 

    ada_clf = AdaBoostClassifier( 
        DecisionTreeClassifier(max_depth = 1), n_estimators = 200,
        algorithm = "SAMME.R", learning_rate = 0.5
    )
    ada_clf.fit(X_train, y_train)    

7.3 Gradient Boosting

    # sequentially fit by the residual errors
    from sklearn.ensemble import GradientBoostingRegressor 

    gdrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
    gdrt.fit(X, y)

7.3 Earlystopping 

    # Stochastic Gradient Boosting 
    # early stopping

    gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True) 

    min_val_error = float("inf")
    error_going_up = 0 
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators 
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val) 
        val_error = mean_squared_error(y_val, y_pred) 
        if val_error < min_val_error: 
            min_val_error = val_error
            error_going_up = 0 
        else:
            error_going_up += 1
            if error_going_up == 5:
                break     


### >>> CH8 Dimensionality Reduction

>>> 8.1 Main Approaches

8.1.1 Projection and Manifold Learning

>>> 8.2 PCA

8.2.1 PCA with sklearn

    from sklearn.decomposition import PCA 
    
    # specify the dimension
    pca = PCA(n_components = 2) 
    X2D = pca.fit_transform(X) 
    
    # explained var ratio
    print(pca.explained_variance_ratio_)
    
    # specify the var ratio to get the right dimensions
    pca == PCA(n_components=0.95) 
    X_reduced = pca.fit_transform(X)
    
    # it is useful to plot and see var ratio
    
    # decompress the data 
    pca = PCA(n_components=154)
    X_reduced = pca.fit_transform(X_train)
    X_recovered = pca.inverse_transform(X_reduced)
   
8.2.2 Speed up PCA

    rnd_pca = PCA(n_components=154, svd_solver="randomized")
    X_reduced = rnd_pca.fit_transform(X)

>>> 8.3 Kernel PCA

8.3.1 KernelPCA (Good at processing cluters data)

    from sklearn.decomposition import KernelPCA
    # different kernels
    lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
    rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
    sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)
    X_reduced = rbf_pca.fit_transform(X)
    
8.3.2 Selecting the Kernel

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline 
    
    # pipeline
    clf = Pipeline([
        ("kpca", KernelPCA(n_components = 2)), 
        ("log_reg", LogisticRegression())
    ])

    # grid search
    param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ['rbf', 'sigmoid']
    }]

    # fit
    grid_search = GridSearchCV(clf, param_grid, cv = 3)
    grid_search.fit(X,y)    
    
    # check the best param 
    print(grid_search.best_params_)

8.3.3 Loally Linear Embedding (LLE)

    from sklearn.manifold import LocallyLinearEmbedding 

    lle = LocallyLinearEmbedding(n_components = 2, n_neighbors = 10) 
    X_reduced = lle.fit_transform(X)

### >>> CH9 

### >>> CH10 ANNs with Keras 

>>> 10.1 From Biological to Artificial Neurons 

>>> 10.2 Implementing MLPs with Keras

10.2.1 Define the model by a list 

    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

10.2.2 Get info about the model 

    # see layers
    model.layers
    
    # see summary and plot the model 
    model.summary()
    keras.utils.plot_model(model, 'name', show_shapes = True) 
    
    # see weights and biases of a layer
    weights, biases = hidden1.get_weights()

10.2.3 Compile and fit the model 

    # compile 
    model.compile(loss = '', optimizer = '', metrics = [])
    
    # fit 
    history = model.fit(X_train, y_train, epochs = , validation_data = (X_valid, y_valid)）

10.2.4 Evaluation and Prediction 

    # plot the learning curve 
    import pandas as pd
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    save_fig("keras_learning_curves_plot")
    plt.show()

    # evaluate and predict
    model.evaluate(X_test, y_test) 
    y_proba = model.predict(X_new) 

10.2.5 Saving and Restoring 

    # normal save and use
    model.save('my_keras_model.h5') 
    model = keras.models.load_model('my_keras_model.h5')

10.2.6 Using Callbacks During Training 

    model.compile() 
    
    # save checkpoint
    checkpoint_cb = keras.callbacks.ModelCheckpoint('my_keras.model.h5',
    save_best_only = True) 
    
    # early stopping 
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
    restore_best_weights=True)
    
    history = model.fit( , , callbacks = [checkpoint_cb,early_stopping_cb])
    
    # rollback to the best model 
    model = keras.models.load_model("my_keras_model.h5")

10.2.7 Model visualization 

    # Tensorboard 
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

10.2.8

>>> 10.3 Fine-tuning Neural Network

10.3.1 Random Search 

    from scipy.stats import reciprocal
    from sklearn.model_selection import RandomizedSearchCV

    param_distribs = {
        "n_hidden": [0, 1, 2, 3],
        "n_neurons": np.arange(1, 100)               .tolist(),
        "learning_rate": reciprocal(3e-4, 3e-2)      .rvs(1000).tolist(),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3,verbose=2)
    rnd_search_cv.fit(X_train, y_train, epochs=100,
                      validation_data=(X_valid, y_valid),
                      callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    # show the best param 
    rnd_search_cv.best_params_
    rnd_search_cv.best_score_
    rnd_search_cv.best_estimator_.model

### >>> CH11 Training Deep Neural Networks

>>> 11.1 Vanishing Gradients 

![image](https://user-images.githubusercontent.com/77312114/134591438-f80dc698-9913-46d0-9dab-18a08c1bdcd3.png)

11.1.1 Initialier and Activation Functions 

    keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")
    

11.1.2 Batch Norm

    keras.layers.BatchNormalization()

11.1.3 Gradient Cliping 

    optimizer = keras.optimizers.SGD(clipvalue=1.0) 

>>> 11.2 Reusing Pretrained Layers 

11.2.1 Save, Clone, and Reuse 

    # save model A 
    model_A.save("my_model_A.h5")
    
    # first clone the model 
    model_A_clone = keras.models.clone_model(model_A) 
    model_A_clone.set_weights(model_A.get_weights())
    
    # reuse model A 
    model_A_clone = keras.models.load_model("my_model_A.h5")
    model_B_on_A = keras.models.Sequential(model_A_clone.layers[:-1])
    model_B_on_A.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # avoid cold-start 

>>> 11.3 Optimizers 

11.3.1 momentum and nesterov gradient

    optimizer = keras.optimizers.SGD(lr = 0.001, momentum = 0.9)
    optimizer = keras.optimizers.SGD(lr = 0.001, momentum = 0.9, nesterov = True)

11.3.2 RMSProp

    optimizer = keras.optimizers.RMSprop(lr = 0.001, tho = 0.9) 

11.3.3 Adam optimization

    optimizer = keras.optimizers.Adam(lr = 0.001, beta_l = 0.9, beta_2 = 0.99)

11.3.4 Learning rate scheduling 

    # power scheduling 
    # exponential scheduling 
    # piecewise constant scheduling 
    # performance sheduling 

>>> 11.4 Regularization

11.4.1 L1 & L2 

    kernel_regularizer=keras.regularizers.l2(0.01)

11.4.2 Use partial to wrap a dense layer 

    from functools import partial

    RegularizedDense = partial(keras.layers.Dense,
                                activation="elu",
                                kernel_initializer="he_normal",
                                kernel_regularizer=keras.regularizers.l2(0.01))

11.4.3 Dropout 

    keras.layers.Dropout(rate = 0.2)
    # remember to adjust output 
    # have also, Monte Carlo Dropout
    # max - norm

11.4.4 A default DNN configuration 

![image](https://user-images.githubusercontent.com/77312114/134591605-d4c8652a-9a37-4610-b934-918ce362e2a8.png)




### >>> CH12 Custom Models and Training with Tensorflow 

>>> 12.1 Using like Numpy

12.1.1 
    
![image](https://user-images.githubusercontent.com/77312114/134591354-360101f3-df20-4de4-bfcf-3dabf873c910.png)

    tf.constant([],[])
    tf.shape()
    tf.Variable()

>>> 12.2 Customizing Models and Training 

12.2.1 Normal Processing 

    # load, split, and scale the data 
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

12.2.2 Define Huber Loss 

    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < 1
        squared_loss = tf.square(error) / 2
        linear_loss = tf.abs(error) - 0.5
        return tf.where(is_small_error, squared_loss, linear_loss)

12.2.3 Custom Layers, custom noise, custom metrics

>>> 12.3 TF Functions and Graphs



### CH13 >>> Loading and Precessing Dara with TensorFlow 

>>> 13.1 The Data API

13.1.1 Data manipulation 

    # Chaining transformations
    dataset = dataset.repeat(3).batch(7)
    
    # create a new one using map 
    dataset = dataset.map(lambda x: x * 2)
    
    # filter the dataset 
    dataset = dataset.filter(lambda x: x < 10)
    
    # shuffle the dataset 
    dataset = tf.data.Dataset.range(10).repeat(3) 
    dataset = dataset.shuffle(buffer_size = 5, seed = 42).batch(7)

13.1.2 Prefetching 

    # Would parallelize threads and GPU

13.1.3 Really large file

    # Split into multiple csv files and pipelining

>>> 13.2 The TFRecord Format

>>> 13.3 Preprocessing the Input Features

13.3.1 One-Hot Encoding

13.3.2 Word Embedding

13.3.3 Keras Preprocessing Layers

    keras.layers.Normzalization()

13.3.4 In-build dataset 

    import tensorflow_datasets as tfds

    # load
    dataset = tfds.load(name = "mnist")

    # train and test
    mnist_train, mnist_test = dataset["train"], dataset["test"]

    # shuffle, batch and prefetch 
    mnist_train = mnist_train.shuffle(10000).batch(32).prefetch(1) 

### CH14 >>> Deep Computer Vision Using Convolutional Neural Networks 

>>> 14.1 Convolutional Layers 

14.1.1 padding: 'SAME' use zero padding, 'VALID' would igore the most outside 

    conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
    padding="same", activation="relu")

>>> 14.2 Pooling Layers 

14.2.1 Max pooling and Avg pooling 

    # usually use max 
    max_pool = keras.layers.MaxPool2D(pool_size = 2) 
    # also have avg pool 
    global_avg_pool = keras.layers.GlobalAvgPool2D() 

>>> 14.3 CNN architectures 

14.3.1 A CNN model 

    model = keras.models.Sequential([
    # 64 * 7 * 7 filters, 28 * 28 pixel 
    keras.layers.Conv2D(64, 7, activation = "relu", padding = "same", 
                       input_shape = [28, 28, 1]),
    keras.layers.MaxPooling2D(2), 
    # increase the number of filters
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"), 
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"), 
    keras.layers.MaxPooling2D(2), 
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2), 
    keras.layers.Flatten(), 
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(64, activation = 'relu'), 
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation = 'relu')
    ])

14.3.2 Other models

    # LeNet-5, AlexNet, GoogleNet, 

14.3 Use pretrained model and transfer learning 

    # download the data 
    #...
    # split and process the data
    #...
    # create the model on top of the Xception model 
    base_model = keras.applications.xception.Xception(weights = "imagenet", include_top = False)
    avg = keras.layers.GlobalAveragePooling2D()(base_model.output) 
    output = keras.layers.Dense(n_classes, activation = "softmax")(avg) 
    model = keras.Model(inputs = base_model.input, outputs = output) 
    
    # freezing the pretrained layers at the beginning 
    for layer in base_model.layers: 
        layer.trainable = False
       #layer.trainable = True

>>> 14.5 Object detection


### >>> CH15 Processing Sequence Using RNNs and CNNs 

>>> 15.1 Recurrent unists and layers


>>> 15.2 Forecasting a Time Series

15.2.1 Create time series data and plot 

15.2.2 Baseline (Linear model and Single RNN)

    Linear_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(1)
    ])
    
    SingleRNN_model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape = [None, 1])
    ])
    
15.2.3 A deep RNN 

    model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences = True, input_shape = [None, 1]),
    keras.layers.SimpleRNN(20, return_sequences = True),
    keras.layers.SimpleRNN(1)
    ])

15.2.4 Predict next 10 steps 

    # train each step predict next 10 steps
    # use TimeDistributed()
    model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences = True, input_shape = [None, 1]),
    keras.layers.SimpleRNN(20, return_sequences = True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])


>>> 15.3 Handling Long Sequences

15.3.1 With batch norm 

    # Usually not helpful in this case
    keras.layers.BatchNormalization()

15.3.2 LSTMs

    keras.layers.LSTM()

15.3.3 GRUs

    keras.layers.GRU()

15.3.4 Using 1D conv layers 

    # As the first layer, to preserve the most and summarize more
    model = keras.models.Sequential([
    keras.layers.Conv1D(filters = 20, kernel_size = 4, strides = 2, padding = 'valid',
                       input_shape = [None, 1]),
    keras.layers.GRU(20, return_sequences = True),
    keras.layers.GRU(20, return_sequences = True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
    
    ])

15.3.5 WaveNet 

    # for extreme long sequence
    # use conv1D with dilation
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[None, 1]))
    for rate in (1, 2, 4, 8) * 2:
        model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
        activation="relu", dilation_rate=rate))
    model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
    


### >>> CH16 NLP with RNNs and Attention

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
    
    
    

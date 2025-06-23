# Week 6: Beyond Linear Models

This week focused on **Beyond Linear Models: Discriminative and Generative Techniques**, covering non-linear, interpretable classifiers and probabilistic methods beyond the limitations of linear decision approaches. The focus was on implementing **Support Vector Machines (SVM)**, **K-Nearest Neighbors (KNN)**, **Naive Bayes**, and **Decision Trees** using real-world datasets and `scikit-learn`.



### Repository Structure

```
data/
    airnb_guest_arrival.csv
post_session/
    hotel_bookings_predicitions.ipynb
pre_session/
    decisionTree.ipynb
    knn_classifier.ipynb
    implementing_svm.ipynb

```

## Pre-Session

The pre-session notebooks introduced the core concepts, mathematical foundations, and practical implementations of four key model types:


### Topics Covered

* **Decision Trees**
  - Impurity metrics: Gini, Entropy
  - Manual tree construction, pruning, and early stopping
  - Overfitting prevention and visual interpretation
  >**Implementation Notebook:** [decisionTree.ipynb](pre_session/decisionTree.ipynb)

* **K-Nearest Neighbors (KNN)**
  - Classification and regression use cases
  - Distance metrics: Euclidean, Manhattan
  - Optimizations with KD-Tree and Ball Tree structures
    
  >**Implementation Notebook:** [knn_classifier.ipynb](pre_session/knn_classifier.ipynb)

* **Naive Bayes**
  - Probabilistic classification using conditional independence
  - Learn to Apply sentiment analysis with text-based features


* **Support Vector Machines (SVM)**
  - Maximum-margin classifiers
  - Kernel methods: linear, polynomial, RBF
  - `SVC` vs `NuSVC`, model tuning using `GridSearchCV`
    
  >**Implementation Notebook:** [implementing_svm.ipynb](pre_session/implementing_svm.ipynb)



## Live Session

* Built decision tree models and benchmarked against linear baselines
* Applied pruning and interpretability techniques using visual tools


## Post-Session
 Used the Hotel Booking Demand dataset to build a Decision Tree Classifier for predicting booking cancellations.

### Task Overview

Using the **Hotel Booking Demand** dataset (`airnb_guest_arrival.csv`), a classification task was performed to predict booking cancellations:

#### Task: Predict `is_canceled` with Decision Tree Classifier

**Steps:**

* Preprocessed data and applied label encoding to categorical features
* Trained a decision tree with `scikit-learn`
* Tuned hyperparameters: `max_depth`, `min_samples_split`
* Compared shallow vs full-depth trees
* Evaluated performance using accuracy, confusion matrix, and classification report

> **Implementation Notebook:**
> [decision\_tree\_classification\_assignment.ipynb](post_session/hotel_bookings_predicitions.ipynb)



## Key Insight

> I explored how SVM, KNN, Naive Bayes, and Decision Trees provide flexible and interpretable alternatives to linear models, each suited to different data types and decision boundaries.


## FUlL FELLOWSHIP JOURNEY: 
> [FellowShip Repo](https://github.com/KushalRegmi61/AI_Fellowship_FuseMachines/tree/master)

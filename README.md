# Week 6: Beyond Linear Models

This week explored non-linear, interpretable, and probabilistic models that extend beyond the limitations of linear approaches. The focus was on implementing **Support Vector Machines (SVM)**, **K-Nearest Neighbors (KNN)**, **Naive Bayes**, and **Decision Trees** using real-world datasets and `scikit-learn`.

---

## Pre-Session

The pre-session notebooks introduced the core concepts, mathematical foundations, and practical implementations of four key model types:

### Repository Structure

```
pre_session/
    decision_tree_basics.ipynb
    knn_modeling.ipynb
    svm_kernels_and_margin.ipynb
data/
    hotel_bookings.csv
```

### Topics Covered

* **Decision Trees**
  - Impurity metrics: Gini, Entropy
  - Manual tree construction, pruning, and early stopping
  - Overfitting prevention and visual interpretation

* **K-Nearest Neighbors (KNN)**
  - Classification and regression use cases
  - Distance metrics: Euclidean, Manhattan
  - Optimizations with KD-Tree and Ball Tree structures

* **Naive Bayes**
  - Probabilistic classification using conditional independence
  - Learn to Apply sentiment analysis with text-based features

* **Support Vector Machines (SVM)**
  - Maximum-margin classifiers
  - Kernel methods: linear, polynomial, RBF
  - `SVC` vs `NuSVC`, model tuning using `GridSearchCV`

---

## Live Session

* Built decision tree models and benchmarked against linear baselines
* Applied pruning and interpretability techniques using visual tools

---

## Post-Session

### Repository Structure

```
post_session/
    decision_tree_classification_assignment.ipynb
```

### Task Overview

Using the **Hotel Booking Demand** dataset (`hotel_bookings.csv`), a classification task was performed to predict booking cancellations:

#### Task: Predict `is_canceled` with Decision Tree Classifier

**Steps:**

* Preprocessed data and applied label encoding to categorical features
* Trained a decision tree with `scikit-learn`
* Tuned hyperparameters: `max_depth`, `min_samples_split`
* Compared shallow vs full-depth trees
* Evaluated performance using accuracy, confusion matrix, and classification report

> **Implementation Notebook:**
> [decision\_tree\_classification\_assignment.ipynb](post_session/decision_tree_classification_assignment.ipynb)



## Key Insight

> I explored how SVM, KNN, Naive Bayes, and Decision Trees provide flexible and interpretable alternatives to linear models, each suited to different data types and decision boundaries.


## FUlL FELLOWSHIP JOURNEY: 
> [FellowShip Repo](https://github.com/KushalRegmi61/AI_Fellowship_FuseMachines/tree/master)

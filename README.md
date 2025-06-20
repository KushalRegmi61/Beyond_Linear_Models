Certainly. Here's your **Week 6: Beyond Linear Models** summary, rewritten in the same structured and professional format as your Week 5 output:

---

### Week 6: Beyond Linear Models

#### Pre-Session Prep

* Decision Tree fundamentals: impurity metrics (Gini, Entropy)
* Handling continuous variables in decision trees
* Manual construction of decision trees from scratch
* Tree inducers and scikit-learn implementation
* Overfitting prevention: early stopping, pruning
* K-Nearest Neighbors (KNN): classification, regression, distance metrics (Brute Force, KD-Tree, Ball Tree)
* Naive Bayes: probabilistic modeling and sentiment analysis
* Support Vector Machines (SVM): margin maximization, kernel methods, C-SVM, ν-SVM variants

#### Live Session Recap

* Applied tree-based models and compared them with linear baselines
* Visualized decision structures for interpretability
* Demonstrated overfitting control via early stopping and pruning
* Explored regression and classification trees on real datasets

#### Post-Session Work

Using a hotel booking demand dataset:

1. **Classification Task:** Predicted booking cancellations (`is_canceled`) using a Decision Tree Classifier

   * Evaluated performance with accuracy, confusion matrix, and classification report

2. **Regression Task:** Estimated guest stay duration using regression trees on `stays_in_number_of_nights`

   * Compared full-depth vs depth-limited models (MSE improved from 5.86 to 4.62)

Key steps:

* Label encoding of categorical features
* Hyperparameter tuning (`max_depth`, `min_samples_split`)
* Early stopping to prevent overfitting
* Performance comparison of shallow vs deep trees

> **Implementation Details:**
> [GitHub – Decision Tree Models](https://github.com/KushalRegmi61/Fusemachines-AI-Fellowship)

#### Key Insight

> I learned how tree-based models, with thoughtful tuning and interpretability, serve as strong and flexible alternatives to linear models for both classification and regression tasks.

---

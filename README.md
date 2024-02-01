# Boosting---Separate-the-Inseparable

This project presents an implementation of Boosting algorithms, with a specific focus on AdaBoost, designed to address challenging classification tasks. By leveraging the power of ensemble learning, the project aims to effectively separate data points initially deemed inseparable by combining multiple weak learners, predominantly decision stumps.

**Implemented Functionalities**

**Decision Stump Implementation:**

The project includes the implementation of the **DecisionStump** class, housed within the **IMLearn/learners/classifiers/decision_stump.py** file. This class is responsible for intelligently selecting a feature from the design matrix X and a corresponding threshold value that yields the optimal classification result when applied solely to that feature.

**AdaBoost Implementation:**

The core of the project lies in the implementation of the **AdaBoost** class, located in the **IMLearn/metalearners/adaboost.py** file. This class serves as the engine behind the AdaBoost algorithm, offering parameterization options for the choice of weak learner (e.g., decision stump) and the number of iterations. Through meticulous orchestration of weak learners, AdaBoost enhances the distribution over misclassified samples in subsequent iterations, ultimately boosting classification accuracy.

**Cross-Validation for Regularization Parameter Selection:**

Additionally, the project encompasses the implementation of cross-validation techniques for selecting regularization parameters using both Lasso and Ridge regressions, particularly relevant when dealing with the diabetes dataset. This aspect of the project involves data preparation, the execution of 5-Fold Cross-Validation for both regularization techniques, and subsequent analysis and interpretation of the results.

**Project Tasks**

**Train AdaBoost Ensemble:**

The project kicks off by training an ensemble of weak learners (e.g., decision stumps) using the AdaBoost algorithm on a dataset devoid of noise. The training and test errors are plotted against the number of fitted learners, offering insights into the algorithm's performance.

**Visualize Decision Boundary:**

To gain further insights into the decision-making process of the AdaBoost ensemble, the project visualizes the decision boundary obtained at different iterations. By overlaying the test set data points on these plots, a comprehensive understanding of the algorithm's performance is provided.

**Optimal Ensemble Size:**

Determining the optimal ensemble size that yields the lowest test error is pivotal. The project plots the decision surface of the optimal ensemble and visualizes the test set data points, thereby providing valuable insights into the ensemble's accuracy.

**Visualize Sample Weights:**

The project visualizes the training set, with point sizes proportional to their weights and color/shapes indicating their labels. Additionally, the decision surface using the full ensemble is plotted, shedding light on the significance of individual samples in the classification process.

**Repeat Analysis with Noise:**

By repeating the analysis for datasets with varying noise levels, the project gains a deeper understanding of the algorithm's robustness and performance under different conditions. The observed behavior of the loss plot is analyzed in terms of the bias-variance tradeoff, offering valuable insights into model performance.

Through a comprehensive exploration of Boosting algorithms and regularization parameter selection using cross-validation, this project significantly contributes to the advancement of ensemble learning and model selection techniques in the field of machine learning.


![Screenshot 2024-02-01 025239](https://github.com/libbyyosef/Machine-Learning---Boosting---Separate-the-Inseparable/assets/36642026/f11c2a59-df88-48f3-9418-696b1b4acf2a)


![Screenshot 2024-02-01 025246](https://github.com/libbyyosef/Machine-Learning---Boosting---Separate-the-Inseparable/assets/36642026/fa6f8d66-f1e1-463e-b7f2-48e588bf3cd5)


![Screenshot 2024-02-01 025304](https://github.com/libbyyosef/Machine-Learning---Boosting---Separate-the-Inseparable/assets/36642026/8dea76bb-3b3f-4d0c-908c-6e2a2aee36da)


![Screenshot 2024-02-01 025310](https://github.com/libbyyosef/Machine-Learning---Boosting---Separate-the-Inseparable/assets/36642026/7eabdbe8-edd6-4955-a689-1c6d402b9992)


![Screenshot 2024-02-01 025317](https://github.com/libbyyosef/Machine-Learning---Boosting---Separate-the-Inseparable/assets/36642026/7f950aa9-7d87-4fae-8c02-58183bd05b28)


![Screenshot 2024-02-01 025324](https://github.com/libbyyosef/Machine-Learning---Boosting---Separate-the-Inseparable/assets/36642026/1a51d1da-4ef1-4619-b735-29f8cc6aa7e3)

![Screenshot 2024-02-01 025329](https://github.com/libbyyosef/Machine-Learning---Boosting---Separate-the-Inseparable/assets/36642026/a088a7ef-4392-4d6e-9634-5612ec704eca)

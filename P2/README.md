# Practical 2

1. Classification

Classification is a supervised machine learning technique used to assign data into predefined categories or classes.

In the loan dataset, classification is used to predict whether a loan application is:

Y → Loan Approved
N → Loan Not Approved

Since there are only two classes, it is called binary classification.

2. Features

Features are the input variables used by a machine learning model to make predictions.

In the loan dataset, examples of features are:

Age
Annual Income
Loan Amount
Credit Score
Employment Years
Education
Marital Status
Property Area
Self Employed

These features are used by the model to predict the loan approval status.

3. Target Variable

The target variable is the variable that the machine learning model tries to predict.

In this dataset:

Target = LoanApproved

It contains:

Y → Loan Approved
N → Loan Not Approved
4. Label Encoding

Label Encoding is a preprocessing technique used to convert categorical/text values into numerical values.

Machine learning algorithms require numerical input, so values such as:

Graduate
Not Graduate

are converted into numbers.

Similarly:

Y → 1
N → 0

This allows the algorithm to process categorical information.

5. Decision Tree

A Decision Tree is a supervised machine learning algorithm used for classification and regression.

It works like a flowchart. It repeatedly divides the dataset based on feature values until it reaches a final prediction.

For example:

Credit Score > 650?
       /       \
     Yes        No
     /           \
 Income > 50K   Not Approved
    /    \
  Yes     No
  /        \
Approved   Not Approved

The final nodes of the tree provide the prediction.

6. Entropy

Entropy is a measure of impurity, uncertainty, or randomness in a dataset.

In a Decision Tree, entropy is used to determine how mixed the classes are.

The formula is:

$$ Entropy = -\sum p_i \log_2(p_i) $$

where \(p_i\) is the probability of each class.

Important points:
Entropy = 0 → dataset contains only one class; completely pure.
Higher entropy → classes are more mixed.
The Decision Tree tries to create splits that reduce entropy.

For example:

100 Approved
0 Not Approved

Entropy = 0

But:

50 Approved
50 Not Approved

Entropy is high.

7. Information Gain

Information Gain measures how much the entropy decreases after splitting the data.

It helps the Decision Tree choose the best feature for splitting.

The basic idea is:

$$ Information\ Gain = Entropy(parent) - Weighted\ Entropy(children) $$

A feature that gives a higher information gain generally provides a better split.

In your code:
criterion="entropy"

means the Decision Tree uses entropy-based splitting.

8. Training Data

Training data is the portion of the dataset used to teach the machine learning model.

In your code, 80% of the data is used for training.

For 600 records:

480 records → Training

The model learns patterns between the features and the target using this data.

9. Testing Data

Testing data is the portion of the dataset that is not used during training.

It is used to evaluate how well the trained model performs on unseen data.

In your code:

120 records → Testing

10. Train-Test Split

Train-test split divides the dataset into training and testing portions.

Your code uses:

test_size=0.20

Therefore:

80% → Training
20% → Testing

This allows us to train the model on one portion and evaluate it on another.

11. Confusion Matrix

A confusion matrix is a table used to evaluate the performance of a classification model.

For binary classification, it contains four values:

	Predicted Negative	Predicted Positive
Actual Negative	TN	FP
Actual Positive	FN	TP

For your loan dataset:

Positive = Loan Approved
Negative = Loan Not Approved
12. True Positive (TP)

True Positive occurs when the actual class is positive and the model correctly predicts positive.

In your case:

Actual = Loan Approved
Predicted = Loan Approved

So the prediction is correct.

13. True Negative (TN)

True Negative occurs when the actual class is negative and the model correctly predicts negative.

In your case:

Actual = Loan Not Approved
Predicted = Loan Not Approved

So the prediction is correct.

14. False Positive (FP)

False Positive occurs when the actual class is negative but the model predicts positive.

In your case:

Actual = Loan Not Approved
Predicted = Loan Approved

This is an incorrect prediction.

15. False Negative (FN)

False Negative occurs when the actual class is positive but the model predicts negative.

In your case:

Actual = Loan Approved
Predicted = Loan Not Approved

This is an incorrect prediction.

16. Accuracy

Accuracy measures the overall proportion of correctly classified observations.

The formula is:

$$ Accuracy = \frac{TP+TN}{TP+TN+FP+FN} $$

For example, if the model correctly predicts 81 out of 120 applicants:

$$ Accuracy = \frac{81}{120}=67.5\% $$

A higher accuracy generally indicates better overall classification performance.

17. Error

Error represents the proportion of incorrectly classified observations.

It can be calculated as:

$$ Error = 1 - Accuracy $$

or:

$$ Error = \frac{FP+FN}{TP+TN+FP+FN} $$

If accuracy is 67.5%:

$$ Error = 100\%-67.5\%=32.5\% $$

A lower error is better.

18. Recall / Sensitivity

Recall, also called Sensitivity or True Positive Rate, measures how well the model identifies actual positive cases.

The formula is:

$$ Recall = \frac{TP}{TP+FN} $$

For your dataset, it tells us:

Out of all applicants whose loans were actually approved, how many were correctly identified as approved?

Higher recall means fewer actual positive cases are missed.

19. Specificity

Specificity measures how well the model identifies actual negative cases.

The formula is:

$$ Specificity = \frac{TN}{TN+FP} $$

For your dataset, it tells us:

Out of all applicants whose loans were actually not approved, how many were correctly identified as not approved?

Higher specificity means fewer negative cases are incorrectly classified as positive.

20. Precision

Although it isn't printed in your current output, it is useful for understanding F1-score.

Precision measures how many of the cases predicted as positive were actually positive.

The formula is:

$$ Precision = \frac{TP}{TP+FP} $$

For your dataset:

Out of all applicants predicted as Loan Approved, how many were actually approved?

21. F1-Score

F1-score is a performance measure that combines precision and recall.

It is the harmonic mean of precision and recall.

$$ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

F1-score ranges from 0 to 1.

1 → perfect performance
0 → very poor performance

It is particularly useful when we want a balance between precision and recall.

22. AUC

AUC stands for Area Under the ROC Curve.

It measures how well a classification model can distinguish between the two classes.

In your case:

Loan Approved
       vs
Loan Not Approved

AUC generally ranges from 0.5 to 1 for a useful classifier:

AUC	General interpretation
0.5	Random classification
0.6–0.7	Poor/weak
0.7–0.8	Fair
0.8–0.9	Good
0.9–1.0	Excellent

Your AUC of around 0.67 indicates moderate/weak discrimination.

23. ROC Curve

ROC stands for Receiver Operating Characteristic.

It is a graph showing the relationship between:

True Positive Rate (Recall)
False Positive Rate

at different classification thresholds.

The area under this curve is called AUC.

24. Cross-Validation

Cross-validation is a technique used to evaluate how well a machine learning model performs on different portions of the dataset.

It reduces dependence on just one train-test split and provides a more reliable estimate of model performance.

25. 5-Fold Cross-Validation

In 5-fold cross-validation, the dataset is divided into five parts.

The model is trained and tested five times.

Round 1 → Test Fold 1, Train on Folds 2–5
Round 2 → Test Fold 2, Train on Folds 1,3–5
Round 3 → Test Fold 3, Train on Folds 1,2,4,5
Round 4 → Test Fold 4, Train on Folds 1–3,5
Round 5 → Test Fold 5, Train on Folds 1–4

The five accuracies are then averaged to obtain the Mean Cross-Validation Accuracy.

26. StratifiedKFold

StratifiedKFold is a version of K-fold cross-validation used for classification.

It tries to maintain a similar proportion of each class in every fold.

For your dataset, it maintains the proportion of:

Loan Approved
Loan Not Approved

across the five folds.

27. Mean CV Accuracy

Mean CV Accuracy is the average accuracy obtained from all five folds.

For example:

Fold 1 = 68%
Fold 2 = 71%
Fold 3 = 77%
Fold 4 = 63%
Fold 5 = 61%

The average of these values gives the Mean CV Accuracy.

It provides an estimate of the model's general performance across different subsets of the data.

28. Prediction Probability

predict_proba() gives the probability of each class.

For example:

Applicant 1

Not Approved = 0.20
Approved     = 0.80

The model predicts:

Loan Approved

because the probability of approval is higher.

These probabilities are also used to calculate AUC.


| Term                         | Theory                                                                                             |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| **Confusion Matrix**         | A table that compares the model's **actual results with predicted results**.                       |
| **TP (True Positive)**       | Actual loan is **approved** and model predicts **approved**.                                       |
| **TN (True Negative)**       | Actual loan is **not approved** and model predicts **not approved**.                               |
| **FP (False Positive)**      | Actual loan is **not approved**, but model predicts **approved**.                                  |
| **FN (False Negative)**      | Actual loan is **approved**, but model predicts **not approved**.                                  |
| **Accuracy**                 | Percentage of **total predictions that are correct**.                                              |
| **Error**                    | Percentage of **total predictions that are incorrect**.                                            |
| **Recall / Sensitivity**     | Measures how well the model identifies **actual approved loans**.                                  |
| **Specificity**              | Measures how well the model identifies **actual not-approved loans**.                              |
| **F1-Score**                 | A combined measure of **precision and recall**.                                                    |
| **AUC**                      | Measures how well the model **distinguishes between approved and not-approved loans**.             |
| **Fold Accuracy**            | Accuracy obtained from each individual fold during cross-validation.                               |
| **Mean CV Accuracy**         | Average accuracy across all **5 folds**, giving a more reliable estimate of model performance.     |
| **Loan Approval Prediction** | The final classification made by the model: **Loan Approved (Yes)** or **Loan Not Approved (No)**. |



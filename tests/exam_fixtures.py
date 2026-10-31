"""Synthetic test data for a fictional 'Machine Learning' course.

This is TEST DATA ONLY. It exists so the engine can be exercised end to end; no
analysis result in exam_engine is derived from, or hardcoded against, these strings.
Records use Member 1's agreed chunk format.
"""

def _p(doc_id, name, page, text):
    return {"document_id": doc_id, "document_name": name, "page_number": page, "text": text}


PAPERS = [
    _p("paper_2022", "ML Past Paper 2022.pdf", 1, """DEPARTMENT OF COMPUTER SCIENCE - FINAL EXAMINATION 2022
Time Allowed: 3 Hours      Maximum Marks: 60
Instructions: Attempt all questions.
Section A (3 marks each)
1. Define overfitting in machine learning.
2. What is the learning rate in gradient descent?
3. State the difference between supervised and unsupervised learning.
Section B
Q4. Explain the working of a neural network. [10 marks]
Q5. (a) Describe how a decision tree chooses a split using information gain. (5)
    (b) Explain pruning of decision trees. (5)"""),
    _p("paper_2022", "ML Past Paper 2022.pdf", 2, """Q6. Discuss the K-means clustering algorithm with an example. [10 marks]
Q7. Explain the bias-variance tradeoff. [10 marks]"""),
    _p("paper_2023", "ML Past Paper 2023.pdf", 1, """DEPARTMENT OF COMPUTER SCIENCE - FINAL EXAMINATION 2023
Time Allowed: 3 Hours      Maximum Marks: 60
Section A (3 marks each)
1. What is overfitting? How can it be avoided?
2. Define regularization.
3. What is a confusion matrix?
Section B
Q4. Describe how artificial neural networks function. [10 marks]
Q5. Explain gradient descent and its variants. [10 marks]
Q6. Describe the K-means clustering procedure and its limitations. [10 marks]"""),
    _p("paper_2023", "ML Past Paper 2023.pdf", 2, """Q7. What is a support vector machine? Explain the role of the kernel. [10 marks]"""),
    _p("paper_2024", "ML Past Paper 2024.pdf", 1, """DEPARTMENT OF COMPUTER SCIENCE - FINAL EXAMINATION 2024
Time Allowed: 3 Hours      Maximum Marks: 60
Section A (3 marks each)
1. Define precision and recall.
2. What is overfitting and why does it occur?
3. Explain the term backpropagation.
Section B
Q4. Explain how decision trees are built and pruned. [10 marks]
Q5. Discuss the training of a neural network using backpropagation. [10 marks]
Q6. Explain the bias and variance trade-off in model selection. [10 marks]
Q7. Explain principal component analysis for dimensionality reduction. [10 marks]"""),
    _p("paper_2025", "ML Past Paper 2025.pdf", 1, """DEPARTMENT OF COMPUTER SCIENCE - FINAL EXAMINATION 2025
Time Allowed: 3 Hours      Maximum Marks: 60
Section A (3 marks each)
1. Define overfitting and explain one technique to reduce it.
2. What is a confusion matrix?
3. Write a short note on regularization.
Section B
Q4. Explain the working of a neural network and its activation functions. [10 marks]
Q5. Describe the K-means algorithm. How is the number of clusters chosen? [10 marks]
Q6. Explain support vector machines and kernel functions. [10 marks]
Q7. Explain gradient descent and the effect of the learning rate. [10 marks]"""),
]

SYLLABUS_TOPICS = [
    "Neural networks",
    "Gradient descent",
    "Decision trees",
    "K-means clustering",
    "Overfitting and regularization",
    "Support vector machines",
    "Bias-variance tradeoff",
    "Principal component analysis",
    "Reinforcement learning",
    "Evaluation metrics",
]

LECTURES = [
    _p("lec_01", "Lecture 3 - Neural Networks.pdf", 4,
       "A neural network is built from layers of artificial neurons. Each neuron computes a weighted sum of its inputs and passes it through an activation function such as ReLU or sigmoid. Information flows forward through hidden layers to the output layer."),
    _p("lec_01", "Lecture 3 - Neural Networks.pdf", 5,
       "Training a neural network uses backpropagation to compute gradients of the loss with respect to every weight. The weights are then updated using gradient descent. The learning rate controls the size of each update; too large a learning rate makes training diverge."),
    _p("lec_02", "Lecture 5 - Trees and Ensembles.pdf", 2,
       "A decision tree repeatedly splits the data on the feature that gives the largest information gain, measured with entropy. Trees that grow too deep memorise the training data, so pruning removes branches that do not improve validation accuracy."),
    _p("lec_03", "Lecture 6 - Clustering.pdf", 1,
       "K-means clustering partitions data into k groups. Centroids are initialised, every point is assigned to its nearest centroid, and centroids are recomputed until assignments stop changing. The elbow method helps choose k."),
    _p("lec_04", "Lecture 2 - Model Generalisation.pdf", 3,
       "A model that fits noise in the training set but fails on new data is said to overfit. Regularization such as L2 weight decay penalises large weights, and early stopping and cross-validation also help models generalise."),
    _p("lec_04", "Lecture 2 - Model Generalisation.pdf", 6,
       "Model error decomposes into bias and variance. High-bias models underfit while high-variance models are too sensitive to the training sample, so choosing model complexity is a trade-off between the two."),
    _p("lec_05", "Lecture 7 - Kernel Methods.pdf", 2,
       "The SVM finds the separating hyperplane with the maximum margin. Kernel functions let it handle data that is not linearly separable by implicitly mapping points to a higher-dimensional space."),
    _p("lec_06", "Lecture 8 - Evaluation.pdf", 1,
       "Accuracy alone can be misleading on imbalanced data. The confusion matrix counts true and false positives and negatives, from which precision and recall are computed."),
    _p("lec_07", "Lecture 9 - Dimensionality.pdf", 1,
       "Dimensionality reduction compresses features while keeping most of the variance. Principal component analysis projects data onto orthogonal directions."),
]
# NOTE: nothing in LECTURES covers reinforcement learning (intentional gap), and the PCA
# lecture is only one short paragraph with no worked example.

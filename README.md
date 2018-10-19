# Neural_Net
Multiclass classification through Neural Net

Anuran Calls (MFCCs) Data Set is available at UCI Dataset Repository(https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29).
Dataset consists of Acoustic features extracted from syllables of anuran (frogs) calls belonging to Acoustic features extracted from syllables of anuran (frogs) calls. But in this analysis only species label is used for classification. Dataset consiste of toal 10 species, and has 7195 instances which were split 4:1 train test ratio.

Neural Net consists of three hiddenlayes (25,15 and 10 neurons in that order) with hyperbolic tangent function activation, followed by softmax layer used for output labels. The model uses Batch Gradient descent with a batch size of 64, for learning cross entropy cost function. On 100 iterations model achieved an accuracy of 0.996 on train set and 0.978 on test set. 

![Alt Text](Performance.png?raw=true "Model Performance")


References:
Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Libraries Used:
Pandas: Version 0.23.3
Tensorflow: Version 1.9.0
Numpy: Version 1.14.5
Matplotlib: Version 2.2.2

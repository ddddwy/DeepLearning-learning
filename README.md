# DeepLearning-learning
>My course notes for the Deep Learning with Python MEAP V06.

1. The machine learning workflow:
* Define the problem and assemble a dataset;
* Define the metrics and choose the loss function:
		balanced classfication problems: accuracy, ROC-AUC;
		class-imbalanced problems: Precision-Recall;
		ranking problems or multi-label classfication: Mean Average Precision.
* Decide on an evaluation protocol:
		hold-out validation set: when you have plenty of data;
		k-fold cross-validation: when you have too few samples;
		iterated k-fold validation: when little data is avaliable.
* Data preprocessing:
		formatted as tensors;
		scaled to small values ([-1,1] range or [0,1] range);
		normalized the heterogenous data;
		feature engineering for small data problems.
* Build a model:
		choice of the last-layer activation;
		choice of loss function;
		choice of optimization configuration: optimizer, learning rate.
* Develop a model that overfits:
		Add layers;
		Add hidden units;
		Train for more epochs.
* Regularize the model and tune its hyperparameters (based on performance on the validation data):
		Add dropout;
		Add or remove layers;
		Add L1 or L2 regularization;
		Try different hyperparameters (the number of units per layer, the learning rate of the optimizer);
		Optionally iterate on feature engineering: add new features, remove uninformative features.
		

2. The most common ways to prevent overfitting in neural networks:
* Get more training data.
* Reduce the capacity of the network.
* Add weight regularization.
* Add dropout.


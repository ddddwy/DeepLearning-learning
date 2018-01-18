# DeepLearning-learning
>My course notes for the Deep Learning with Python MEAP V06.

1. The machine learning workflow:<br>
* Define the problem and assemble a dataset;
* Define the metrics and choose the loss function:<br>
		balanced classfication problems: accuracy, ROC-AUC;<br>
		class-imbalanced problems: Precision-Recall;<br>
		ranking problems or multi-label classfication: Mean Average Precision.
* Decide on an evaluation protocol:<br>
		hold-out validation set: when you have plenty of data;<br>
		k-fold cross-validation: when you have too few samples;<br>
		iterated k-fold validation: when little data is avaliable.
* Data preprocessing:<br>
		formatted as tensors;<br>
		scaled to small values ([-1,1] range or [0,1] range);<br>
		normalized the heterogenous data;<br>
		feature engineering for small data problems.
* Build a model:<br>
		choice of the last-layer activation;<br>
		choice of loss function;<br>
		choice of optimization configuration: optimizer, learning rate.<br>
* Develop a model that overfits:<br>
		Add layers;<br>
		Add hidden units;<br>
		Train for more epochs.
* Regularize the model and tune its hyperparameters (based on performance on the validation data):<br>
		Add dropout;<br>
		Add or remove layers;<br>
		Add L1 or L2 regularization;<br>
		Try different hyperparameters (the number of units per layer, the learning rate of the optimizer);<br>
		Optionally iterate on feature engineering: add new features, remove uninformative features.
		

2. The most common ways to prevent overfitting in neural networks:<br>
* Get more training data.
* Reduce the capacity of the network.
* Add weight regularization.
* Add dropout.


# DeepLearning-learning
>My course notes for the Deep Learning with Python MEAP V06.

1. The machine learning workflow:
* Define the problem and assemble a dataset;
* Define the metrics and choose the loss function:
	* balanced classfication problems: accuracy, ROC-AUC;
	* class-imbalanced problems: Precision-Recall;
	* ranking problems or multi-label classfication: Mean Average Precision.
* Decide on an evaluation protocol:
	* hold-out validation set: when you have plenty of data;
	* k-fold cross-validation: when you have too few samples;
	* iterated k-fold validation: when little data is avaliable.
* Data preprocessing:
	* formatted as tensors;
	* scaled to small values ([-1,1] range or [0,1] range);
	* normalized the heterogenous data;
	* feature engineering for small data problems.
* Build a model:
	* choice of the last-layer activation;
	* choice of loss function;
	* choice of optimization configuration: optimizer, learning rate.
* Develop a model that overfits:
	* Add layers;
	* Add hidden units;
	* Train for more epochs.
* Regularize the model and tune its hyperparameters (based on performance on the validation data):
	* Add dropout;
	* Add or remove layers;
	* Add L1 or L2 regularization;
	* Try different hyperparameters (the number of units per layer, the learning rate of the optimizer);
	* Optionally iterate on feature engineering: add new features, remove uninformative features.

2. The most common ways to prevent overfitting in neural networks:
* Get more training data.
* Reduce the capacity of the network.
* Add weight regularization.
* Add dropout.

3. Deep learning for computer vision:
* Fight overfitting in small dataset: use visual data augmentation;
* Achieve higher accuracy: use a pre-trained convnet to do feature extraction and fine-tuning;
* Represent the learning process: generate visualizations of the outputs, filters and heatmaps of the convnet.

4. Pseudo-code simple RNN:
```python
state_t=0
for input_t in input_sequence:
	output_t=activation(dot(W, input_t)+dot(U, state_t)+b)
	state_t=output_t
```
A RNN is just a 'for' loop that reuses quantities computed during the previous iteration of the loop.<br>

5. Pseudo-code LSTM:
```python
output_t=activation(dot(state_t, Uo)+dot(input_t, Wo)+dot(C_t, Vo)+bo)

i_t=activation(dot(state_t, Ui)+dot(input_t, Wi)+bi)
f_t=activation(dot(state_t, Uf)+dot(input_t, Wf)+bf)
k_t=activation(dot(state_t, Uk)+dot(input_t, Wk)+bk)

c_t+1=i_t*k_t + c_t*f_t 
```
The LSTM cell is meant to allow past information to be reinjected at a later time, thus fighting the vanishing gradient problem.<br>
The strength of LSTM is in question answering and machine translation.<br>

6. 1D convnet for processing sequences:
* Consist of stacks of Conv1D layers and MaxPooling1D layers, and eventually end in a global pooling operation or flattening operation.
* Use a 1D convnet as a preprocessing step before a RNN, shortening the sequence and extracting useful representations for the RNN to process.

7. Build high-performing deep convnets:
* Residual connections
* Batch normalization (BatchNormalization())
* Depthwise separable convolutions (SeparableConv2D())

8. Hyperparameter optimization:
* Hyperopt library or Hyperas library.
* Be careful about validation set overfitting!



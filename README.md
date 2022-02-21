# Kaggle-Natural-Language-Processing-with-Disaster-Tweets
Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).In this competition, we’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. The dataset cosnists of 10,000 tweets that were hand classified.

# Approach
The tweets were tokenized and encoded using BERT. Optinally, the tweet embeddings are passed through a bi-directional LSTM on top of BERT for further encoding. Finally, the tweet enbeddings are given as input to a Multi Layer Perceptron (MLP) binary classification head. The two neural architectures used are depicted in the following diagramms.

# Model evaluation and optimization
The models were evaluated using 4-fold cross validation. Hyperparameter tuning was conducted using Bayesian optimization. The optimization results follow.
<p align="center"><img src="https://github.com/NickKaparinos/Kaggle-Natural-Language-Processing-with-Disaster-Tweets/blob/master/images/contour.png" alt="drawing" width="1300"/></p>
<p align="center"><img src="https://github.com/NickKaparinos/Kaggle-Natural-Language-Processing-with-Disaster-Tweets/blob/master/images/param_importances.png" alt="drawing" width="1300"/></p>

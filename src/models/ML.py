import numpy as np
from src.utils import utils
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


def encode(data):
	"""
	Encode the different classes with sklearn and remove associate
	columns in the training and testing dataset
	LabelEncoder : http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

	Args:
		data: Pandas Dataframe
			Dataframe after preprocessing (standardization, filling blanks)
	Returns:
		Dataframe encoded, labels and columns
	"""
	df = data[data.columns.drop(list(data.filter(regex='pred|flag')))]
	predicted_labels = data.filter(regex='pred|flag')
	le = LabelEncoder()
	le.fit(df.True_Label)
	labels = le.transform(df.True_Label)  # encode species strings
	columns = list(le.classes_)  # save column names for submission
	data_output = df.drop(['ID', 'True_Label'], axis=1)
	return data_output, labels, columns, predicted_labels, le


def mean_log(log_mean):
	"""

	Args:
		log_mean:

	Returns:

	"""
	cl = pd.DataFrame(log_mean.Classifier.unique(), columns=['Classifier'])
	log_mean = log_mean.groupby(log_mean.index).mean()
	final_log = log_mean.merge(cl, left_index=True, right_index=True)
	columns = final_log.columns.tolist()
	columns.insert(0, columns.pop(-1))
	final_log = final_log[columns]
	final_log = final_log.sort_values(by=['Accuracy'], ascending=False)
	final_log = final_log.reset_index()
	final_log.index += 1
	del final_log['index']
	return final_log


def cut_off(target, predicted):
	""" Find the optimal probability cutoff point for a classification model related to event rate
	Parameters
	----------
	target : Matrix with dependent or target data, where rows are observations

	predicted : Matrix with predicted data, where rows are observations

	Returns
	-------
	list type, with optimal cutoff value

	"""
	fpr, tpr, threshold = metrics.roc_curve(target, predicted)
	i = np.arange(len(tpr))
	roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i),
	                    'threshold': pd.Series(threshold, index=i)})
	roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]

	return list(roc_t['threshold'])


def compute_metrics(train_predictions, train_predictions_proba, y_test, name):
	"""

	Args:
		name:
		train_predictions:
		train_predictions_proba:
		y_test:

	Returns:

	"""
	log_cols = ["Classifier", "Accuracy", "Error", "Sensitivity", "Specificity", "False Positive Rate",
	            "Precision", "Matthews Correlation Coefficient", "F1 Score", "Average Precision Score", "Log Loss",
	            "AUC Score", ]
	confusion = metrics.confusion_matrix(y_test, train_predictions)
	# TN, FP, FN, TP = metrics.confusion_matrix(y_test, train_predictions).ravel()
	# print(metrics.confusion_matrix(y_test, train_predictions).ravel())
	# exit()
	# exit()
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]

	# ACCURACY
	acc = round(metrics.accuracy_score(y_test, train_predictions), 4)

	# ERROR
	err = 1 - acc

	# SENSITIVITY
	ss = round(TP / float(TP + FN), 4)
	# print(ss)

	# SPECIFICITY
	sc = round(TN / float(TN + FP), 4)

	# FALSE POSITIVE RATE
	fpr = round(FP / float(TN + FP), 4)

	# PRECISION
	pr = round(TP / float(TP + FP), 4)

	# MATTHEWS CORRELATION COEFFICIENT
	mcc = round(metrics.matthews_corrcoef(y_test, train_predictions), 4)

	# F1-SCORE
	f_one = round(metrics.f1_score(
		y_test, train_predictions, average="macro"), 4)

	# AVERAGE PRECISION SCORE
	apr = round(metrics.average_precision_score(
		y_test, train_predictions_proba), 4)

	# LOG LOSS
	ll = round(metrics.log_loss(y_test, train_predictions), 3)

	# AUC SCORE METRICS

	auc = round(metrics.roc_auc_score(y_test, train_predictions_proba), 4)

	log_entry = pd.DataFrame([[name, acc, err, ss, sc, fpr,
	                           pr, mcc, f_one, apr, ll, auc]], columns=log_cols)

	log = log_entry.reset_index(drop=True)

	return log


def adjusted_classes(y_scores, t):
	"""
	This function adjusts class predictions based on the prediction threshold (t).
	Will only work for binary classification problems.
	"""
	return [1 if y >= t else 0 for y in y_scores]


def dump_models(classifiers, output_dir):
	"""

	Args:
		classifiers:
		output_dir:

	Returns:

	"""
	utils.mkdir(output_dir + "/Models")
	for clf in classifiers:
		try:
			best_clf = clf.best_estimator_
		except AttributeError:
			best_clf = clf
		clf_name = "/Models/" + best_clf.__class__.__name__
		joblib.dump(best_clf, output_dir + clf_name + '.pkl')


def dump_prediction(results_proba, results_pred, output_dir, data):
	complete_matrix = data.reset_index(drop=True)
	for algo in results_proba:
		tmp_df_proba = pd.DataFrame(
			data=results_proba[algo], columns=[algo + '_proba'])
		tmp_df_pred = pd.DataFrame(
			data=results_pred[algo], columns=[algo + '_pred'])
		tmp_df = pd.concat([tmp_df_proba, tmp_df_pred], axis=1)
		complete_matrix = pd.concat([complete_matrix, tmp_df], axis=1)

	complete_matrix.to_csv(output_dir + '/Complete_matrix_with_predictions.csv.gz',
	                       sep='\t', compression='gzip', index=False)

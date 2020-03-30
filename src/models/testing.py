from sklearn.externals import joblib
from src.utils import utils
from src.models import ML
from src.visualization import visualization
import glob
import os
import pandas as pd
import numpy as np
import warnings


class TestingClassification(object):
	def __init__(self,
	             input_data,
	             output_dir,
	             model_dir,
	             standardize,
	             logger,
	             threshold
	             ):
		"""
        Init
        Args:
            input_data:
            output_dir:
            model_dir:
            logger:
        """
		warnings.filterwarnings('ignore')

		utils.mkdir(output_dir)
		self.logger = logger
		self.logger.info('\n')
		self.logger.info('=' * 100)
		self.logger.info('You will TEST the trained model on selected data : {}'
		                 .format(os.path.basename(input_data)))
		self.logger.info('=' * 100)
		self.logger.info('\n')
		df = utils.prepare_input_data(input_data=input_data,
		                              standardize=standardize,
		                              )
		df = df.reset_index(drop=True)
		logger.info('TESTING on {} samples'.format(df.shape[0]))
		if model_dir.endswith("Models"):
			model = model_dir
		elif 'Model' in model_dir:
			model = model_dir
		else:
			model = model_dir + "/TRAIN/Models"
		classifiers = self.load_classifiers(model_dir=model)
		output_dir = output_dir + "/TEST"
		utils.mkdir(output_dir)
		self.launch(data=df,
		            classifiers=classifiers,
		            output_dir=output_dir,
		            threshold=threshold)

	@staticmethod
	def load_classifiers(model_dir):
		"""

        Args:
            model_dir:

        Returns:

        """
		# os.chdir(model_file)
		classifiers = dict()
		for mod in glob.glob(model_dir + "/*.pkl"):
			sk_model = joblib.load(mod)
			classifiers[os.path.basename(mod).replace('.pkl', '')] = sk_model
		return classifiers

	@staticmethod
	def find_nearest(array, value):
		array = np.asarray(array)
		idx = (np.abs(array - value)).argmin()
		return array[idx]

	def launch(self, data, classifiers, output_dir, threshold):
		"""
        Launch classifiers evaluation and allow to save output results

        Args:
            threshold:
            data: Pandas Dataframe
                Dataframe with the preprocessed data corresponding to the selected mode
                (complete data, selected predictors)
            classifiers: list
                List of classifiers tested
            output_dir: str
                Name of the output directory

        Returns: None

        """
		self.logger.info('Encoding data...')
		encode_data, labels, classes, predicted_labels, le = ML.encode(data)

		log_list = list()
		results_proba = dict()
		results_pred = dict()
		predictors = list(data[data.columns.drop(list(data.filter(regex='pred|flag')))].drop(['ID', 'True_Label'], axis=1).columns)
		fi = pd.DataFrame()

		for name, clf in classifiers.items():
			train_predictions_proba = clf.predict_proba(encode_data.values)
			pred_proba = train_predictions_proba[:, 1]

			pred_adj = ML.adjusted_classes(pred_proba, threshold)
			tmp_log = ML.compute_metrics(pred_adj, pred_proba, labels, name)
			results_pred[name] = le.inverse_transform(pred_adj)
			log_list.append(tmp_log)
			results_proba[name] = pred_proba

			# Getting feature weights if RF, LR or VC combining both
			tmp_fi = pd.DataFrame()
			if name == 'VotingClassifier':
				tmp_fi_rf = pd.Series(clf.named_estimators_['rf'].feature_importances_, predictors).sort_values(ascending=False).to_frame()
				tmp_fi_lr = pd.Series(clf.named_estimators_['lr'].coef_[0], predictors).sort_values(ascending=False).to_frame()
				tmp_fi_rf.columns = ['RandomForestClassifier_VC']
				tmp_fi_lr.columns = ['LogisticRegression_VC']
				tmp_fi_rf = (100. * tmp_fi_rf / tmp_fi_rf.sum()).round(2)
				tmp_fi_lr = (100. * tmp_fi_lr.abs() / tmp_fi_lr.abs().sum()).round(2)
				tmp_fi = pd.concat([tmp_fi_rf, tmp_fi_lr], axis=1)
			if name == 'RandomForestClassifier':
				tmp_fi = pd.Series(clf.feature_importances_, predictors).sort_values(ascending=False).to_frame()
				tmp_fi.columns = ['RandomForestClassifier']
				tmp_fi = (100. * tmp_fi / tmp_fi.sum()).round(2).astype(str) + '%'
			elif name == 'LogisticRegression':
				tmp_fi = pd.Series(clf.coef_[0], predictors).sort_values(ascending=False).to_frame()
				tmp_fi.columns = ['LogisticRegression']
				tmp_fi = (100. * tmp_fi.abs() / tmp_fi.abs().sum()).round(2).astype(str) + '%'
			fi = pd.concat([fi, tmp_fi], axis=1)
		fi.to_csv(output_dir + '/Feature_importances.csv', sep='\t', index=True)

		log_mean = pd.concat(log_list)
		log = log_mean.sort_values(by=['Accuracy'], ascending=False)
		log = log.reset_index(drop=True)

		self.logger.info('Dumping labels and proba predictions ...')
		ML.dump_prediction(results_proba, results_pred, output_dir, data)
		visualization.print_stdout(self.logger, log, output_dir, predicted_labels, labels, data)
		self.logger.info('Saving plots and results...')
		visualization.plot_roc_curve_testing(labels, results_proba, output_dir, predicted_labels, data, ylim=-0.05)
		visualization.plot_roc_curve_testing(labels, results_proba, output_dir, predicted_labels, data, ylim=0.95)

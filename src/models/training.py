import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from src.utils import utils
from src.models import ML
from src.visualization import visualization
import collections
import os
import pandas as pd
import warnings


class TrainingClassification(object):

	def __init__(self,
	             input_data,
	             output,
	             classifiers,
	             standardize,
	             logger,
	             cv,
	             plot=True,
	             ):
		"""
        Init method for Classification class

        Parameters
        -----------
        plot : bool
            If enable, save graphs and subplots in the output directory
        predictors : list
            List of predictors present in the header of the dataset, default=Complete table
        standardize : bool
            If enable, standardize the dataframe (mu=0, std=1) with StandardScaler() (see scikit-learn)
        split_columns : bool
            If enable, allows to split columns in the dataframe
            check : http://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
        classifiers : list
            list of specific classifiers selected to test on the dataset, default = GaussianNB, LogisticRegression.
            Complete list : 'MLPClassifier, KNeighborsClassifier,
            SVC, NuSVC, DecisionTreeClassifier, RandomForestClassifier,
            AdaBoostClassifier, GradientBoostingClassifier, GaussianNB,
            LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, LogisticRegression')
        output : str
            output: Output_directory, default = current directory
        input_data : String reference the input file (CSV format)
            path to the input file
        full_clf : bool
            Enable test of all available Classification Algorithms

        """
		utils.mkdir(output)
		starttime = datetime.now()
		self.logger = logger
		self.logger.info('Processing of input data'.format(
			os.path.splitext(input_data)[0]))
		print('\n')
		print('=' * 100)
		self.logger.info('You will TRAIN outputs on selected data : {}'
		                 .format(os.path.splitext(input_data)[0]))
		print('=' * 100)
		print('\n')

		df = utils.prepare_input_data(input_data=input_data,
		                              standardize=standardize,
		                              )

		pd.set_option('display.float_format', lambda x: '%.3f' % x)

		logger.info('TRAINING on {} samples'.format(df.shape[0]))
		output = output + "/TRAIN"
		self.launch(data=df,
		            classifiers=classifiers,
		            output_dir=output,
		            plot=plot,
		            cv=cv)

		endtime = datetime.now()
		self.logger.info("Script duration : " +
		                 str(endtime - starttime).split('.', 2)[0])

	def warn(*args, **kwargs):
		pass

	warnings.warn = warn

	def launch(self, data, classifiers, output_dir, plot, cv):
		"""
        Launch classifiers evaluation and allow to save output results

        Args:
            cv:
            data: Pandas Dataframe
                Dataframe with the preprocessed data corresponding to the selected mode
                (complete data, selected predictors)
            classifiers: list
                List of classifiers tested
            output_dir: str
                Name of the output directory
            plot: bool
                If enable, save the different results plots into the output directory
        Returns: None

        """
		self.logger.info('Encoding data...')
		encode_data, labels, classes, predicted_labels, le = ML.encode(data)
		results_proba, dict_y_test, classifiers = self.stratifier(
			encode_data, labels, classifiers, cv, output_dir)
		self.logger.info('Saving outputs...')
		ML.dump_models(classifiers, output_dir)
		if plot is True:
			utils.mkdir(output_dir + "/Plots")
			self.logger.info('Saving plots and results...')
			visualization.plot_roc_curve_training(
				dict_y_test, results_proba, output_dir)

	def stratifier(self, data, labels, classifiers, cv, output_dir):
		"""
        1. Split data in training and testing set
        2. Send split data to the classification method to evaluate algorithms
        3. Group log results to report a mean log with all the metrics produced

        StratifiedShuffleSplit method of sklearn is used :
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

        Args:
            output_dir:
            cv:
            data: Pandas dataframe
                Data encoded in the encode method
            labels: numpy array
                Encode classes to represent labels
            classifiers: list
                List of classifiers to compare

        Returns:
            final_log : Pandas Dataframe
                Final mean log with all metrics computed
            results_prob : dict
                Probabilites calculated for each test in a dict
            dict_y_test : dict
                Dict which is save all the y_test used in each evaluation
        """
		results_proba = collections.defaultdict(dict)
		dict_y_test = collections.defaultdict()
		sss = StratifiedShuffleSplit(n_splits=cv, test_size=0.2, random_state=3)
		sss.get_n_splits(data, labels)
		i = 1
		self.logger.info('Training processing ...')
		loop = sss.split(data, labels)
		t = tqdm(loop)
		l = collections.defaultdict(dict)
		for train_index, test_index in t:
			t.set_description('Cross-validation n°')
			x_train, x_test = data.values[train_index], data.values[test_index]
			y_train, y_test = labels[train_index], labels[test_index]
			dict_y_test[i] = y_test
			results_proba, tmp_l = \
				self.classification(
					i, classifiers, results_proba, x_train, x_test, y_train, y_test)
			[l[d].update(tmp_l[d]) for d in tmp_l]
			i += 1
		[l[clf].update({'Mean': np.mean(np.asarray(list(l[clf].values())))})
		 for clf in l]
		log_cv = pd.DataFrame(l)
		log_cv.index.names = ['Cross-validation']
		log_cv.to_csv(output_dir + '/Cross-validation_accuracy.csv',
		              index=True, sep='\t')
		print('Cross-validation results : \n')
		print(log_cv)

		return results_proba, dict_y_test, classifiers

	@staticmethod
	def classification(i, classifiers, results_proba, x_train, x_test, y_train, y_test):
		"""
        Core method which is compare all the algorithms and where metrics are computed

        Args:
            y_test:
            i:
            classifiers: list
                List of all classifiers evaluated
            results_proba: dict
                Empty dict which will be saved all the computed proba results
            x_train: numpy array
                Training set
            x_test: numpy array
                Testing set
            y_train: numpy array
                Training labels
        Returns:
            results_proba : dict
                Probabilites computed

        """
		tmp_l = dict()
		for clf in classifiers:
			clf.fit(x_train, y_train)
			try:
				best_clf = clf.best_estimator_
			except AttributeError:
				best_clf = clf
			name = best_clf.__class__.__name__
			train_predictions_proba = clf.predict_proba(x_test)
			train_predictions_proba = train_predictions_proba[:, 1]
			results_proba[name].update({i: train_predictions_proba})
			tmp_l[name] = {str(i): best_clf.score(x_test, y_test)}
		return results_proba, tmp_l

import pathlib

from src.models.testing import TestingClassification
from src.models.training import TrainingClassification
from src.utils import utils
from src.features import select_columns_pandas
from src.evaluation import combination_pandas
from src.visualization import histo_weights, maf_plot

from datetime import datetime
from sklearn.externals import joblib
import argparse
import glob
import numpy as np
import os
import pandas as pd
import sys
import warnings
import parmap

# Sklearn algorithms import
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

# Supress sklearn warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("always")

# Project Description

__project__ = 'MISTIC '
__author__ = 'Thomas'
__maintainer__ = 'Thomas'
__email__ = 'thomas.weber@etu.unistra.fr'


def test_eval_mp(file):
	if file.endswith('.csv.gz'):
		fname = file.replace('.csv.gz', '').split('_')
		cv = fname[0]
		maf_test = fname[1]
		file_name = "_".join(fname[2:])
		logger.disabled = True
		output_dir = arg_dict['output_dir']
		output_dir = output_dir.split('/')
		output_dir[-1] = 'EVALUATION_SETS_' + output_dir[-1]
		output_dir = "/".join(output_dir)
		if os.path.exists(output_dir + '/RESULTS_' + file_name + '_' + maf_test + '_' + cv) is False:
			TestingClassification(input_data=output_dir + '/' + file,
			                      standardize=arg_dict['standardize'],
			                      output_dir=output_dir + '/RESULTS_' + file_name + '_' + maf_test + '_' + cv,
			                      model_dir=arg_dict['model'],
			                      logger=logger,
			                      threshold=arg_dict['threshold']
			                      )


# 3 ways to use MISTIC_toolbox : TRAINING + TESTING, TESTING, PREDICTION (no stats or plots, just predicted proba + labels)

# First one and most important : TRAINING AND TESTING on a given dataframe

def training_and_testing(ARGS):
	# Check conditions
	if ARGS['list_columns']:
		list_columns = list(sorted(ARGS['list_columns']))
	if not ARGS['list_columns']:
		list_columns = ['CADD_phred', 'SIFTval', 'VEST4_score', 'gnomAD_exomes_AF']

	if ARGS['flag']:
		flag = list(sorted(ARGS['flag']))
	if not ARGS['flag']:
		flag = ["REVEL_score", "ClinPred_score", "M-CAP_score", "fathmm-XF_coding_score", "Eigen-raw_coding",
		        "PrimateAI_score", ]


	if not os.path.exists(ARGS['output_dir'] + '/TRAIN/training.csv.gz') or not os.path.exists(
			ARGS['output_dir'] + '/TEST/testing.csv.gz'):
		logger.warn(
			'--train_and_test mode selected but training and testing file not found, creation with following parameters :'
			'--ratio : ' + str(ARGS['ratio']) + ', --proportion : ' + str(ARGS['proportion']))
		ARGS['force_datasets'] = True
	if os.path.exists(ARGS['output_dir'] + '/TRAIN/training.csv.gz') or os.path.exists(
			ARGS['output_dir'] + '/TEST/testing.csv.gz'):
		logger.info('Training and testing file found')

	if ARGS['combinatory'] is True:
		pass

	# if enable, erase previously generated training and testing file from a global dataframe to creating other ones
	if ARGS['force_datasets'] is True:

		utils.mkdir(ARGS['output_dir'])
		utils.mkdir(ARGS['output_dir'] + '/TRAIN')
		utils.mkdir(ARGS['output_dir'] + '/TEST')
		logger.warn('Creating new files or overwriting old ones')
		prop = ARGS['proportion']
		t = float(round(prop / (1 - prop), 2))

		ratio = ARGS['ratio']
		tmp = pd.read_csv(filepath_or_buffer=ARGS['input'], sep='\t', compression='gzip', encoding='utf-8', low_memory=False)


		if list_columns and flag:
			# Selection of specific columns to be used from a global dataframe
			# Example : df with 10 columns, --list_columns column1 column2 column5
			tmp = select_columns_pandas.select_columns_pandas(tmp, list_columns, flag)
			logger.info(tmp)

		# Use of input parameters to build training and testing dataframes (proportion, ratio of data between train and test)
		# Special attention is paid to remove overlap between evaluation|test sets and training dataset to prevent any overfitting

		complete_data_path = tmp.loc[tmp['True_Label'] == 1]
		complete_data_path = complete_data_path.sample(frac=1)
		complete_data_begn = tmp.loc[tmp['True_Label'] == -1]
		complete_data_begn = complete_data_begn.sample(frac=1)
		max_size = max(complete_data_path.shape[0], complete_data_begn.shape[0])
		min_size = min(complete_data_path.shape[0], complete_data_begn.shape[0])


		if max_size > (t * min_size):
			max_size = min_size * t
		elif max_size < (t * min_size):
			min_size = max_size / t
		if min_size < 1000 and min(complete_data_path.shape[0], complete_data_begn.shape[0]) == \
				complete_data_path.shape[0]:
			logger.warn(
				'CAREFUL : Size of the pathogenic dataset will be < 1000 samples')



		eval_test_size = ratio
		train_path = complete_data_path.head(
			n=int(round(min_size * (1 - eval_test_size))))
		train_begn = complete_data_begn.head(
			n=int(round(max_size * (1 - eval_test_size))))
		eval_path = complete_data_path.tail(
			n=int(round(min_size * eval_test_size)))
		eval_begn = complete_data_begn.tail(
			n=int(round(min_size * eval_test_size)))

		eval_path.dropna(inplace=True)
		eval_begn.dropna(inplace=True)

		complete_training = pd.concat([train_path, train_begn]).drop_duplicates(keep='first')
		complete_training = complete_training[
			complete_training.columns.drop(list(complete_training.filter(regex='pred|flag')))]
		complete_training.dropna(inplace=True)

		# Some stats on Pathogenic and Benign variant numbers in both training and testing dataframes

		logger.info('Training - Path : ' + str(complete_training[complete_training['True_Label'] == 1].shape[0]))
		logger.info('Training - Benign : ' + str(complete_training[complete_training['True_Label'] == -1].shape[0]))

		min_size_eval = min(eval_path.shape[0], eval_begn.shape[0])
		complete_eval = pd.concat([eval_path.sample(frac=1).head(min_size_eval),
		                           eval_begn.sample(frac=1).head(min_size_eval)]).drop_duplicates(keep='first')

		logger.info('Testing - Path : ' + str(complete_eval[complete_eval['True_Label'] == 1].shape[0]))
		logger.info('Testing - Benign : ' + str(complete_eval[complete_eval['True_Label'] == -1].shape[0]))

		# Dumping data

		complete_training.to_csv(path_or_buf=ARGS['output_dir'] + '/TRAIN/training.csv.gz',
		                         sep='\t',
		                         compression='gzip',
		                         encoding='utf-8',
		                         index=False)

		complete_eval.to_csv(path_or_buf=ARGS['output_dir'] + '/TEST/testing.csv.gz',
		                     sep='\t',
		                     compression='gzip',
		                     encoding='utf-8',
		                     index=False)

	check_dir_train = False
	if os.path.isdir(ARGS['output_dir'] + '/TRAIN/Models'):
		check_dir_train = True
	if (ARGS['force_training'] is True) or (check_dir_train is False):
		# Training model
		TrainingClassification(input_data=ARGS['output_dir'] + '/TRAIN/training.csv.gz',
		                       classifiers=classifiers,
		                       standardize=ARGS['standardize'],
		                       output=ARGS["output_dir"],
		                       logger=logger,
		                       cv=ARGS['cross_validation']
		                       )

		TestingClassification(input_data=ARGS['output_dir'] + '/TEST/testing.csv.gz',
		                      standardize=ARGS['standardize'],
		                      output_dir=ARGS["output_dir"],
		                      model_dir=ARGS['model'],
		                      logger=logger,
		                      threshold=ARGS['threshold']
		                      )

		# Generation of a histogram to see most important features used in builded model
		histo_weights.histo_and_metrics(folder=ARGS['output_dir'], logger=logger)

	# This parameter, if enabled, will build all possible combinations from a single dataframe if sources are mentionned
	# Example : A global dataframe based on 3 databases (2 pathogenic : Clinvar and HGMD and 1 benign : gnomAD) was generated
	# The following lines will generate 2 evaluation sets : (clinvar|gnomAD) and (HGMD|gnomAD) with various MAF thresholds (<0.01, <0.001, 0.0001, AC=1(singleton), AF=0)
	# and each of these combinations will be tested with the previously generated outputs. (Overlapping is checked between these combinations and training dataset)

	if ARGS['eval'] and ARGS['eval'].endswith('.csv.gz'):
		# TODO : CHANGE NAME
		print('\n\n')
		logger.info('--BUILDING & TESTING ON EVALUATION SETS--')

		output_dir = ARGS['output_dir']
		eval_output_dir = output_dir
		eval_output_dir = eval_output_dir.split('/')
		eval_output_dir[-1] = 'EVALUATION_SETS_' + eval_output_dir[-1]
		eval_output_dir = "/".join(eval_output_dir)

		if os.path.isdir(eval_output_dir):
			pass
		else:
			utils.mkdir(eval_output_dir)

			# if ARGS['list_columns'] and ARGS['flag']:
			combination_pandas.combination_pandas(ARGS['eval'], output_dir + '/TRAIN/training.csv.gz',
			                                      eval_output_dir, logger, list_columns, flag,
			                                      CV=ARGS['cross_validation_evaluation'])
		# else:
		# 	combination_pandas.combination_pandas(ARGS['eval'], ARGS['output_dir'] + '/TRAIN/training.csv.gz', output_dir, CV=ARGS['cross_validation_evaluation'])
		l_dir = os.listdir(eval_output_dir)

		parmap.starmap(test_eval_mp, list(zip(l_dir)), pm_pbar=True, pm_processes=ARGS['threads'])

	# Plots are automatically generated to visualize performance across various scenario for the different combinations
	print('\n\n')
	logger.info('--GENERATING PLOTS & STATS--')
	utils.mkdir(eval_output_dir + '/PLOTS_AND_MEAN_TABLE')
	maf_plot.violin_plot_scores(eval_output_dir, logger)
	maf_plot.maf_plot_maf_0(eval_output_dir, ARGS['cross_validation_evaluation'], logger)
	maf_plot.maf_plot_others(eval_output_dir, ARGS['cross_validation_evaluation'], logger)


# maf_plot.combine_maf_plot('outputs/EVALUATION_SETS_' + ARGS['output_dir'] + '/PLOTS_AND_MEAN_TABLE')


# Second one : testing previously generated model on some data (require a dataframe and a model)
def testing(ARGS):
	list_columns = list(sorted(ARGS['list_columns']))
	flag = list(sorted(ARGS['flag']))
	tmp = select_columns_pandas.select_columns_pandas(
		pd.read_csv(ARGS['input'], compression='gzip', sep='\t', low_memory=False), list_columns, flag)
	input_modified = ARGS['input'].replace('.csv.gz', '_lite.csv.gz')
	tmp.to_csv(input_modified, compression='gzip', sep='\t', index=False)
	TestingClassification(input_data=input_modified,
	                      standardize=ARGS['standardize'],
	                      output_dir=ARGS["output_dir"],
	                      model_dir=ARGS['model'],
	                      logger=logger,
	                      threshold=ARGS['threshold'],
	                      )


# This last mode is only used to predict proba and label on some variants without performing metrics analyses or plot
# Input : Just a dataframe with the same features used in training
# Output : the same dataframes with predicted probabilities and predicted labels for each algorithm tested


def prediction(ARGS):
	return_df = True
	# BASIC
	list_columns = list(sorted(ARGS['list_columns']))
	flag = list(sorted(ARGS['flag']))
	input_file = ARGS['input']
	output_file = input_file.replace('.csv.gz', '_MISTIC.csv.gz')

	output_dir = ARGS['output_dir']
	model_dir = ARGS['model']
	select = ARGS['wt_select']

	utils.mkdir(output_dir)

	# IMPORT DF
	data = pd.read_csv(filepath_or_buffer=input_file, sep='\t', compression='gzip', encoding='utf-8', low_memory=False)
	data['ID'] = data['ID'].str.lstrip('chr_')

	# SELECT GOOD COLUMNS
	if select is True:
		data = select_columns_pandas.select_columns_pandas(data, list_columns, flag, progress_bar=False, fill=True, dropna=False)
		col_ordered = ['ID', 'True_Label'] + list(sorted(set(list(data.columns)) - set(['ID', 'True_Label'])))
		data = data[col_ordered]
	if select is False:
		data = data[list_columns + flag]

	data['True_Label'] = data['True_Label'].replace(-1, 0)

	if 'Amino_acids' in list_columns:
		l_cols = [e for e in list_columns if e != 'Amino_acids']
	else:
		l_cols = list_columns

	data_scoring = data.dropna(subset=l_cols)

	# IMPORT SKLEARN MODELS
	classifiers = dict()
	log = list()
	for mod in glob.glob(model_dir + "/*.pkl"):
		sk_model = joblib.load(mod)
		classifiers[os.path.basename(mod).replace('.pkl', '')] = sk_model
		name = os.path.basename(mod).replace('.pkl', '')
		data_scoring[name + '_proba'] = sk_model.predict_proba(data_scoring[l_cols])[:, 1]
		data_scoring[name + '_pred'] = sk_model.predict(data_scoring[l_cols])
		data = pd.concat([data, data_scoring[[name + '_proba', name + '_pred']]], axis=1)

	col_ordered = ['ID', 'True_Label'] + list(sorted(set(list(data.columns)) - set(['ID', 'True_Label'])))
	data = data[col_ordered]
	with_maf = data[data['gnomAD_exomes_AF'] != 0]
	without_maf = data[data['gnomAD_exomes_AF'] == 0]
	data['MISTIC_pred'] = pd.concat([with_maf['MISTIC_VC_pred'], without_maf['MISTIC_LR_pred']], axis=0).sort_index()
	data['MISTIC_proba'] = pd.concat([with_maf['MISTIC_VC_proba'], without_maf['MISTIC_LR_proba']], axis=0).sort_index()
	data.drop(['MISTIC_VC_pred', 'MISTIC_VC_proba', 'MISTIC_LR_pred', 'MISTIC_LR_proba'], axis=1, inplace=True)


	if return_df is False:
		data.to_csv(output_file, compression='gzip', index=False, sep='\t')
	elif return_df is True:
		return data



if __name__ == "__main__":

	text = "Welcome to MISTIC's Toolbox"

	print('=' * 30)
	print(text)
	print('=' * 30)

	starttime = datetime.now()

	# Argument parsing to get all the options required or not to use MISTIC toolbox
	parser = argparse.ArgumentParser(
		description='Toolbox designed to build ML outputs based on Scikit-Learn. Used to build MISTIC and mMISTIC outputs',
		usage='%(prog)s [--help]')

	required = parser.add_argument_group('Required arguments')

	mode = parser.add_argument_group('MISTIC modes')

	optional = parser.add_argument_group('MISTIC options')

	mode.add_argument('--train',
	                  action='store_true',
	                  help='Train only algorithms on specified data')

	mode.add_argument('--train_and_test',
	                  action='store_true',
	                  help='Train and Test the selected algorithms on specified data')

	mode.add_argument('--test',
	                  action='store_true',
	                  help='Test only algorithms on specified data based on saved outputs')

	mode.add_argument('--prediction',
	                  action='store_true',
	                  help='Predict only scores and labels without additional results')

	required.add_argument('-i', '--input',
	                      metavar='',
	                      type=str,
	                      required=True,
	                      help='Input file')

	optional.add_argument('-o', '--output_dir',
	                      type=str,
	                      metavar='',
	                      help='Output directory, default = Current directory')

	optional.add_argument('-e', '--eval',
	                      type=str,
	                      metavar='',
	                      help='')

	optional.add_argument('-fd', '--force_datasets',
	                      action='store_true',
	                      help='If enable, create new training and testing sets from input data. Training and testing files will be saved in the ouput directory.')

	optional.add_argument('-gs', '--gene_specific',
	                      action='store_true',
	                      help='If enable, create new training and testing sets from input data with respect of genes. Same amount of variants from benign and pathogenic sources will be sampled. Training and testing genes are non redundant. Then Training and testing files will be saved in the ouput directory.')

	optional.add_argument('-c', '--combinatory',
	                      action='store_true',
	                      help='If enable, create new training and testing sets from input data. Training and testing files will be saved in the ouput directory.')

	optional.add_argument('-ft', '--force_training',
	                      action='store_true',
	                      help='If enable, create new training and testing sets from input data. Training and testing files will be saved in the ouput directory.')

	optional.add_argument('-m', '--model',
	                      metavar='',
	                      type=str,
	                      help='Model directory, required if --test mode or --prediction mode enabled')

	optional.add_argument('-prop', '--proportion',
	                      type=float,
	                      metavar='',
	                      default=0.5,
	                      help='If enable, selected proportion will be maintained between classes')

	optional.add_argument('-std', '--standardize',
	                      action='store_true',
	                      help='Standardize data with scikit-learn.preprocessing.StandardScaler, default=False')

	optional.add_argument('--wt_select',
	                      action='store_false',
	                      help='Standardize data with scikit-learn.preprocessing.StandardScaler, default=False')

	optional.add_argument('-fb', '--fill_blanks',
	                      action='store_true',
	                      help='Fill blanks or not, default=False')

	optional.add_argument('-fbs', '--fill_blanks_strategy',
	                      type=str,
	                      metavar='',
	                      default="median",
	                      help='"mean", "median", "most_frequent" or "constant"')

	optional.add_argument('-v', '--vote',
	                      action='store_true',
	                      help='Voting system of RF and LR enabled')

	optional.add_argument('-g', '--grid',
	                      action='store_true',
	                      help='If enable, perform randomized grid search for tuning hyperparameters')

	optional.add_argument('--n_iterations',
	                      type=int,
	                      metavar='',
	                      default=10,
	                      help='Number of n-iteration to perform during randomized grid search, '
	                           'Larger is the number, longer is the script duration, default = 10')

	optional.add_argument('--list_columns',
	                      metavar='',
	                      nargs="*",
	                      type=str,
	                      help='')

	optional.add_argument('--flag',
	                      metavar='',
	                      nargs="*",
	                      type=str,
	                      default=list(),
	                      help='')

	optional.add_argument('-cv', '--cross_validation',
	                      type=int,
	                      metavar='',
	                      default=1,
	                      help='Number of cross-validation to perform, default = 1')

	optional.add_argument('-cve', '--cross_validation_evaluation',
	                      type=int,
	                      metavar='',
	                      default=10,
	                      help='Number of cross-validation to perform during evaluation step, default = 10')

	optional.add_argument('-cv_gs', '--cross_validation_grid_search',
	                      type=int,
	                      metavar='',
	                      default=0,
	                      help='Number of cross-validation to perform during grid search optimization, '
	                           'proportion between classes are conserved, default = 1')

	optional.add_argument('-r', '--ratio',
	                      type=float,
	                      metavar='',
	                      default=0.2,
	                      help='Ratio between training and testing datasets, '
	                           'default = 0.2 of the input file will be assigned to testing set')

	optional.add_argument('-t', '--threshold',
	                      type=float,
	                      metavar='',
	                      default=0.5,
	                      help='Custom threshold to classify binary data')

	optional.add_argument('--threads',
	                      type=int,
	                      metavar='',
	                      default=4,
	                      help='Enable parallelization of specific algorithms (example : LogisticRegression)'
	                           'To use all cores available on your hardware, specify -1'
	                      )

	args = parser.parse_args()

	arg_dict = vars(args)

	# All the parameters you want to try if you used the grid mode

	voting_parameters = {
		# "rf__class_weight" : {0:1,1:19},
		"rf__max_depth": [1, 2, 3, None],
		"rf__min_samples_split": [2, 3, 10],
		"rf__bootstrap": [True, False],
		"rf__criterion": ["gini", "entropy"],
		'rf__n_estimators': [10, 50, 100, 200],
		'lr__solver': ['lbfgs', 'saga'],
		'lr__tol': [0.1, 0.01, 0.001, 0.0001],
		'lr__penalty': ['l2'],
		'lr__C': np.logspace(0, 4, 10),
		'lr__max_iter': [100, 500, 1000, 2000, 4000],
		# 'lr__class_weight' : {0:1,1:19},
	}

	parameters_dict = {
		'MLPClassifier': {'solver': ['lbfgs'],
		                  'max_iter': [500, 1000, 1500],
		                  'alpha': 10.0 ** -np.arange(1, 7),
		                  'hidden_layer_sizes': np.arange(1, 12),
		                  'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
		                  # 'verbose' : True
		                  },
		'RandomForestClassifier': {"max_depth": [1, 2, 3, None],
		                           "min_samples_split": [2, 3, 10],
		                           "bootstrap": [True, False],
		                           "criterion": ["gini", "entropy"],
		                           'n_estimators': [10, 50, 100, 200],
		                           # 'verbose': True
		                           },
		'LogisticRegression': {'solver': ['lbfgs', 'saga'],
		                       'penalty': ['l2'],
		                       'C': np.logspace(0, 4, 10),
		                       'max_iter': [100, 500, 1000, 2000, 4000],
		                       'tol': [0.1, 0.01, 0.001, 0.0001]
		                       # 'verbose': True,
		                       },
		'GradientBoostingClassifier': {"loss": ["deviance"],
		                               "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
		                               "min_samples_split": np.linspace(0.1, 0.5, 12),
		                               "min_samples_leaf": np.linspace(0.1, 0.5, 12),
		                               "max_depth": [3, 5, 8],
		                               "max_features": ["log2", "sqrt"],
		                               "criterion": ["friedman_mse", "mae"],
		                               "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
		                               "n_estimators": [10],
		                               # 'verbose': True,
		                               },
		'LinearDiscriminantAnalysis': {'solver': ['svd', 'lsqr', 'eigen', ],
		                               # 'verbose': True,
		                               },
		'VotingClassifier': voting_parameters

	}

	# Classifiers list

	classifiers = [
		VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier(n_estimators=100))],
		                 voting='soft'),
		LogisticRegression(n_jobs=arg_dict['threads'], max_iter=1000),
		RandomForestClassifier(n_estimators=100, n_jobs=arg_dict['threads']),
		# MLPClassifier(),
		# GradientBoostingClassifier(n_estimators=100),
		# GaussianNB(),
		# LinearDiscriminantAnalysis(),
		# LinearSVC(),
		# DecisionTreeClassifier()
	]

	grid_classifiers = [
		MLPClassifier(),
		LogisticRegression(),
		RandomForestClassifier(n_estimators=100),
		# MLPClassifier(),
		# GradientBoostingClassifier(n_estimators=100),
	]

	# IF GRID ENABLED

	if arg_dict['grid'] is True:
		rdm_clf = list()
		for clf in grid_classifiers:
			rdm_clf.append(
				RandomizedSearchCV(
					clf,
					param_distributions=parameters_dict[clf.__class__.__name__],
					n_iter=arg_dict['n_iterations'],
					cv=arg_dict['cross_validation_grid_search'],
					scoring='roc_auc',
					n_jobs=arg_dict['threads'],
				)
			)

		classifiers = rdm_clf

	logger = utils.setup_custom_logger("Classification", arg_dict['output_dir'])

	# Conditions

	if arg_dict["cross_validation"] > 1 and arg_dict["cross_validation_grid_search"] > 0:
		logger.warning(': both --cross_validation and --cross_validation_grid_search parameters were instanced')

	if arg_dict["output_dir"] is None:
		cwd = os.getcwd()
		output_dir = cwd + '/outputs/OUTPUT_EXAMPLE'
		arg_dict["output_dir"] = output_dir

	if not os.path.exists(arg_dict["output_dir"]) and arg_dict["output_dir"] is not None:
		try:
			pathlib.Path(arg_dict["output_dir"]).mkdir(exist_ok=True)
		except FileNotFoundError:
			logger.error('Unable to find or create this output directory')
			sys.exit("============\nSee you soon :)\n============")

	if arg_dict['input'] is not None:
		test = os.path.isfile(arg_dict['input'])
		if test is True:
			pass
		if test is False:
			logger.error('File not exists')
			sys.exit("============\nSee you soon :)\n============")

	if arg_dict['fill_blanks_strategy'] is not None and arg_dict['fill_blanks_strategy'] not in ['mean', 'median',
	                                                                                             'constant',
	                                                                                             'most_frequent']:
		logger.error('Strategy selected not exists')
		sys.exit("============\nSee you soon :)\n============")

	if arg_dict['model'] is None and arg_dict['train_and_test'] is True:
		arg_dict['model'] = arg_dict["output_dir"]

	# LAUNCHING MODES

	if arg_dict['prediction'] is True:
		prediction(arg_dict)

	elif arg_dict["train_and_test"] is True and (arg_dict["test"] or arg_dict["train"]) is False:
		training_and_testing(arg_dict)

	elif arg_dict["test"] is True and (arg_dict["train_and_test"] or arg_dict["train"]) is False:
		testing(arg_dict)

	elif (arg_dict["train_and_test"] and arg_dict["train"] and arg_dict["test"]) is False:
		logger.error('No mode selected')
		sys.exit("============\nSee you soon :)\n============")

	else:
		logger.error('Two modes selected or input file missing')
		sys.exit("============\nSee you soon :)\n============")

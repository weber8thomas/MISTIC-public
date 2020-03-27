import collections
import seaborn as sns
import matplotlib
from sklearn import metrics

from src.utils import utils
from src.models import ML

matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import operator
import pandas as pd

plt.ioff()

convert_dict = {
	"CADD_phred_pred": "CADD_phred",
	"ClinPred_score_pred": "ClinPred_flag",
	"Condel_pred": "Condel",
	"DEOGEN2_score_pred": "DEOGEN2_flag",
	"PrimateAI_score_pred": "PrimateAI_flag",
	# "DEOGEN2_score_pred" : "DEOGEN2_score",
	"fathmm-XF_coding_score_pred": "fathmm-XF_coding_flag",
	"Eigen-raw_coding_pred": "Eigen-raw_coding_flag",
	"M-CAP_score_pred": "M-CAP_flag",
	"MetaLR_score_pred": "MetaLR_score",
	"MetaSVM_score_pred": "MetaSVM_score",
	"MutationTaster_score_pred": "MutationTaster_flag",
	"PolyPhenVal_pred": "PolyPhenVal",
	"REVEL_score_pred": "REVEL_flag",
	"SIFTval_pred": "SIFTval",
	"VEST4_score_pred": "VEST4_score",
	# 'VotingClassifier' : 'VotingClassifier',
	'VotingClassifier': 'MISTIC',
	'LogisticRegression': 'LogisticRegression',
	'RandomForestClassifier': 'RandomForestClassifier',
	'GradientBoostingClassifier': 'GradientBoostingClassifier',
	'MLPClassifier': 'MLPClassifier',
	'GaussianNB': 'GaussianNB',
	'MISTIC': 'MISTIC'
}

legend_dict = {
	'CADD': 'Integrated',
	'Condel': 'Integrated',
	'MetaLR': 'Integrated',
	'MetaSVM': 'Integrated',
	'PolyPhenVal': 'Integrated',
	'SIFTval': 'Integrated',
	'VEST4': 'Integrated',
	'ClinPred': 'ClinPred',
	'FATHMM': 'Comparators',
	'MutationTaster': 'MutationTaster',
	'fathmm-XF': 'fathmm-XF',
	'DEOGEN2': 'DEOGEN2',
	'PrimateAI': 'PrimateAI',
	'Eigen-raw': 'Eigen-raw',
	'REVEL': 'REVEL',
	'M-CAP': 'M-CAP',
	# 'MISTIC' : 'MISTIC',
	'VotingClassifier': 'MISTIC',
	# 'VotingClassifier' : 'VotingClassifier',
	'LogisticRegression': 'LogisticRegression',
	'RandomForestClassifier': 'RandomForestClassifier',
	'GradientBoostingClassifier': 'GradientBoostingClassifier',
	'MLPClassifier': 'MLPClassifier',
	'GaussianNB': 'GaussianNB',
	'Gnomad_Alone': 'Gnomad_Alone',
	'Clinvar_Alone': 'Clinvar_Alone',
	'gnomAD_global_MAF': 'gnomAD_global_MAF',
	'Multi_ethnics_gnomAD_MAF': 'Multi_ethnics_gnomAD_MAF',
}

color_dict = {
	'Integrated': '#9E9E9E',
	'ClinPred': '#2196F3',
	'M-CAP': '#FF9800',
	'fathmm-XF': '#9b59b6',
	'MetaLR': '#9C27B0',
	'CADD': '#9C27B0',
	'MetaSVM': '#673AB7',
	'REVEL': '#4CAF50',
	'DEOGEN2': '#34495e',
	'PrimateAI': '#22a6b3',
	'Eigen-raw': '#0c2461',
	'MISTIC': '#f44336',
	'VotingClassifier': '#f44336',
	'LogisticRegression': '#e57373',
	'RandomForestClassifier': '#2196F3',
	'GradientBoostingClassifier': '#9b59b6',
	'GaussianNB': '#22a6b3',
	'MLPClassifier': '#0c2461',
	'Gnomad_Alone': '#f44336',
	'Clinvar_Alone': '#f44336',
	'gnomAD_global_MAF': '#gnomAD_global_MAF',
	'Multi_ethnics_gnomAD_MAF': '#Multi_ethnics_gnomAD_MAF',
}


def plot_roc_curve_training(dict_y_test, results_proba, output_dir):
	"""
    Method for plot ROC comparison between algorithms

    Args:
        dict_y_test: dict
            Store the y_test used for each iteration of CV
        results_proba : dict
            Store the proba obtained for each iteration of CV for every algorithm used
        output_dir: str
            Directory where will be save the plots
    Returns:
        None
    """

	output_dir = output_dir + '/Plots'
	utils.mkdir(output_dir)
	dict_auc = dict()
	ordered_dict = collections.defaultdict(dict)

	matplotlib.rcParams.update({'font.size': 8})

	for algo, results in results_proba.items():
		fig_algo = figure(num=algo, dpi=180, facecolor='w', edgecolor='k')
		plt.figure(algo)
		tprs = list()
		aucs = list()
		mean_fpr = np.linspace(0, 1, 100)
		for index, arrays in results.items():
			fpr, tpr, thresholds = metrics.roc_curve(dict_y_test[index], arrays)
			tprs.append(np.interp(mean_fpr, fpr, tpr))
			tprs[-1][0] = 0.0
			roc_auc = metrics.auc(fpr, tpr)
			aucs.append(roc_auc)

			plt.plot(fpr, tpr, lw=1, alpha=0.3, label=r'ROC CV n°%s (AUC = %0.2f)' % (index, roc_auc))

		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = metrics.auc(mean_fpr, mean_tpr)

		std_auc = np.std(aucs)
		plt.plot(mean_fpr, mean_tpr, color='red',
		         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
		         lw=2, alpha=.8)
		std_tpr = np.std(tprs, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

		dict_auc[algo] = mean_auc

		ordered_dict[algo]['mean_tpr'] = mean_tpr
		ordered_dict[algo]['mean_fpr'] = mean_fpr
		ordered_dict[algo]['std_auc'] = std_auc
		ordered_dict[algo]['std_tpr'] = std_tpr
		ordered_dict[algo]['tprs_upper'] = tprs_upper
		ordered_dict[algo]['tprs_lower'] = tprs_lower

		plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
		                 label=r'$\pm$ 1 std. dev.')
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating curve : ' + str(algo) + ' | ' + str(index) + ' fold cross-validation')
		plt.legend(loc="lower right")

		plt.savefig(output_dir + "/ROC_" + str(algo) + ".png", bbox_inches='tight', dpi=180)

		plt.close(fig_algo)

	fig_glob = figure(num='Global', dpi=180, facecolor='w', edgecolor='k')
	plt.figure('Global')

	ordered_dict_auc = sorted(dict_auc.items(), key=operator.itemgetter(1), reverse=True)

	for elem in ordered_dict_auc:
		algo = elem[0]
		mean_auc = elem[1]

		try:

			plt.plot(ordered_dict[algo]['mean_fpr'], ordered_dict[algo]['mean_tpr'],
			         label=r'Mean ROC : ' + str(algo) + ' - AUC = %0.2f $\pm$ %0.2f)' % (
				         mean_auc, ordered_dict[algo]['std_auc']),
			         lw=2, alpha=.8)
		except:
			plt.plot(ordered_dict[algo]['mean_fpr'], ordered_dict[algo]['mean_tpr'],
			         label=r'Mean ROC : ' + str(algo) + ' - AUC = %0.2f' % (
				         mean_auc),
			         lw=2, alpha=.8)

		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating curve : Global results | ' + str(index) + ' fold cross-validation')
		plt.legend(loc="lower right")
	plt.savefig(output_dir + "/ROC_Summary.png", bbox_inches='tight', dpi=180)
	plt.close(fig_glob)


def plot_roc_curve_testing(y_test, results_proba, output_dir, predicted_labels, data, ylim):
	"""
    Method for plot ROC comparison between algorithms

    Args:
        ylim:
        data:
        predicted_labels:
        y_test:
        results_proba : dict
            Store the proba obtained for each iteration of CV for every algorithm used
        output_dir: str
            Directory where will be save the plots
    Returns:
        None
    """
	# output_dir = output_dir + '/Plots/ROC_plots'
	dict_auc = dict()
	ordered_dict = collections.defaultdict(dict)

	threshold_sup = dict()

	for algo, results in results_proba.items():
		fpr, tpr, thresholds = metrics.roc_curve(y_test, results_proba[algo])
		roc_auc = metrics.roc_auc_score(y_test, results_proba[algo])

		if ylim == 0.95:
			tmp_tpr = tpr - 0.95
			tmp_tpr[np.where(tmp_tpr < 0)[0]] = 0.0
			roc_auc = metrics.auc(fpr, tmp_tpr) / 0.05
			threshold_sup[algo] = thresholds[-len(tmp_tpr[tmp_tpr != 0]):][0]

		dict_auc[algo] = roc_auc
		ordered_dict[algo]['tpr'] = tpr
		ordered_dict[algo]['fpr'] = fpr

	matplotlib.rcParams.update({'font.size': 8})
	fig_glob = figure(num='Global', dpi=120, facecolor='w', edgecolor='k')
	plt.figure('Global')

	from matplotlib.lines import Line2D

	check = ['CADD', 'M-CAP', 'REVEL', 'ClinPred', 'Condel', 'VEST4', 'MetaLR', 'MetaSVM', 'SIFTval', 'PolyPhenVal',
	         'fathmm-XF', 'DEOGEN2', 'Eigen-raw', 'PrimateAI']

	for prediction in predicted_labels.columns:
		if 'pred' in prediction:
			prediction_score = convert_dict[prediction]
			fpr, tpr, thresholds = metrics.roc_curve(y_test, data[prediction_score])
			roc_auc = metrics.roc_auc_score(y_test, data[prediction_score])

			if ylim == 0.95:
				tmp_tpr = tpr - 0.95
				tmp_tpr[np.where(tmp_tpr < 0)[0]] = 0.0
				roc_auc = metrics.auc(fpr, tmp_tpr) / 0.05
				threshold_sup[prediction] = thresholds[-len(tmp_tpr[tmp_tpr != 0]):][0]

			predictor = prediction.split('_')[0]
			dict_auc[predictor] = roc_auc
			ordered_dict[predictor]['tpr'] = tpr
			ordered_dict[predictor]['fpr'] = fpr

	ordered_dict_auc = sorted(dict_auc.items(), key=operator.itemgetter(1), reverse=False)
	# plt.plot([0, 1], [0, 1], color='black', lw=0.5, linestyle=':')

	for elem in ordered_dict_auc:
		algo = elem[0]
		if algo in check:
			plt.plot(ordered_dict[algo]['fpr'], ordered_dict[algo]['tpr'], color=color_dict[legend_dict[algo]], lw=2)
		if algo == 'VotingClassifier':
			plt.plot(ordered_dict[algo]['fpr'], ordered_dict[algo]['tpr'], color=color_dict[legend_dict[algo]], lw=4)
		plt.xlim([-0.05, 1.05])
		if ylim == 0.95:
			plt.ylim([ylim, 1.00])
		else:
			plt.ylim([ylim, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')

	custom_lines = [
		Line2D([0], [0], color=color_dict['VotingClassifier'], lw=4),
		Line2D([0], [0], color=color_dict['M-CAP'], lw=2),
		Line2D([0], [0], color=color_dict['REVEL'], lw=2),
		Line2D([0], [0], color=color_dict['ClinPred'], lw=2),
		Line2D([0], [0], color=color_dict['Eigen-raw'], lw=2),
		Line2D([0], [0], color=color_dict['fathmm-XF'], lw=2),
		Line2D([0], [0], color=color_dict['PrimateAI'], lw=2),
		# Line2D([0], [0], color=color_dict['Integrated'], lw=2),
	]
	plt.legend(custom_lines, [
		'MISTIC (' + str(round(dict_auc['VotingClassifier'], 3)) + ')',
		'M-CAP (' + str(round(dict_auc['M-CAP'], 3)) + ')',
		'REVEL (' + str(round(dict_auc['REVEL'], 3)) + ')',
		'ClinPred (' + str(round(dict_auc['ClinPred'], 3)) + ')',
		'Eigen (' + str(round(dict_auc['Eigen-raw'], 3)) + ')',
		'FATHMM-XF (' + str(round(dict_auc['fathmm-XF'], 3)) + ')',
		'PrimateAI (' + str(round(dict_auc['PrimateAI'], 3)) + ')',
		# "MISTIC's features",
	]
	           )

	plt.title('ROC Curve on VarTest')
	plt.savefig(output_dir + "/ROC_Evaluation_" + str(ylim) + ".png", bbox_inches='tight', dpi=200)
	plt.clf()
	plt.close()

	if ylim == 0.95:
		dict_threshold_output = pd.DataFrame.from_dict(threshold_sup, orient='index')
		dict_threshold_output.columns = ['Threshold_95%_Sensitivity']
		dict_threshold_output.to_csv(output_dir + '/Threshold_95percent_sensitivity.csv', sep='\t')


def print_stdout(logger, log, output_dir, predicted_labels=None, labels=None, data=None):
	"""
    Print to standard output and save results table to csv and html table

    Args:
        logger:
        predicted_labels:
        labels:
        data:
        log: Pandas Dataframe
            Dataframe with the metrics (ROC-AUC, PR-AUC, Accuracy ...)
            linked to each scenario
        output_dir: str
            Name of the output directory

    Returns:
        None

    """

	if predicted_labels is not None and labels is not None and data is not None:
		tmp_list = list()
		log_list = list()
		for prediction in predicted_labels.keys():
			if 'pred' in prediction and 'FATHMM' not in prediction:
				prediction_score = convert_dict[prediction]
				tmp = data[prediction].values

				tmp = [0 if int(e) == -1 else int(e) for e in tmp]
				name = prediction.split('_')[0]
				l = ML.compute_metrics(tmp, data[prediction_score].values, labels, name)
				l['Classifier'] = prediction.split('_')[0]
				log_list.append(l)
		log_mean = pd.concat(log_list)
		log_mean = log_mean.sort_values(by=['AUC Score'], ascending=False).reset_index(drop=True)
		final_df = pd.concat([log, log_mean], axis=0)
		output_dir = output_dir

		name_html = "/Classification_summary.html"
		name_csv = "/Classification_summary.csv"

		pd.set_option('display.max_colwidth', -1)
		col_ordered = ['Classifier', 'AUC Score'] + list(sorted(set(list(final_df.columns)) - {'Classifier',
																							 'AUC Score'}))
		final_df = final_df[col_ordered]
		final_df.reset_index(drop=True)
		final_df.Classifier = final_df.Classifier.str.replace('-raw', '')
		final_df.Classifier = final_df.Classifier.str.replace('fathmm-XF', 'FATHMM-XF')
		logger.info('\n')
		logger.info('TEST STEP RESULTS : \n')
		logger.info(final_df.reset_index(drop=True))
		final_df.to_csv(str(output_dir) + str(name_csv), sep='\t', index=False)

		utils.write_to_html_file(final_df, title="Benchmark results", filename=str(output_dir) + str(name_html))
		histo_radmel(final_df, output_dir)
		logger.info('=' * 100)
		logger.info('=' * 100)
		logger.info('\n' * 3)



def histo_radmel(df, output):
	sns.set(style="white")
	sns.set_context("paper")
	matplotlib.rcParams['interactive'] == True
	fig, ax = plt.subplots(figsize=(8, 6))
	# df = pd.read_csv(file, sep='\t')
	# df.replace({'VotingClassifier': 'MISTIC'}, inplace=True)
	if 'SPECIFIC' in output:
		print('SPECIFIC')
		df.replace({'VotingClassifier': 'VC', 'LogisticRegression': 'MISTIC', 'RandomForestClassifier': 'RF'},
		           inplace=True)
	else:
		df.replace({'VotingClassifier': 'MISTIC', 'LogisticRegression': 'LR', 'RandomForestClassifier': 'RF'},
		           inplace=True)
	competitors = ['', 'PrimateAI', 'Eigen', 'FATHMM-XF', 'ClinPred', 'M-CAP', 'REVEL', 'MISTIC']
	# competitors = ['M-CAP', 'REVEL', 'ClinPred', 'RF', 'LR', 'MISTIC']
	df = df[df['Classifier'].isin(competitors)]
	df.set_index('Classifier', inplace=True)
	df = df.reindex(competitors)
	df = df.reset_index()
	# df.Classifier = df.Classifier.str.replace('MISTIC', 'hsMISTIC')
	df = df[['Classifier', 'AUC Score', 'F1 Score', 'Log Loss']]

	# df = df[['Classifier', 'Specificity', 'F1 Score', 'Log Loss']]
	# CLINPRED M-CAP REVEL MISTIC
	colors = ["#22a6b3", "#0c2461", "#9b59b6", "#2196F3", "#FF9800", "#4CAF50", "#f44336"]
	# colors = ["#2196F3", "#FF9800", "#4CAF50", "#e57373", "#e57373" ,"#f44336"]

	ax.yaxis.grid()

	# df.plot(kind='bar', x='Classifier', y=['AUC Score', 'F1 Score', 'Log Loss'], secondary_y='Log Loss')
	g1 = df.plot.line(marker='x', linestyle='-', x='Classifier', y='AUC Score', ylim=(0, 1.1), color='r', ax=ax,
	                  label='ROC AUC ', lw=3, markersize=10, fontsize=10)
	g2 = df.plot.line(marker='o', linestyle='-', x='Classifier', y='F1 Score', ylim=(0, 1.1), color='b', ax=ax,
	                  label='F1 Score', lw=3, markersize=10, fontsize=10)
	g3 = df.plot.line(marker='s', linestyle='--', x='Classifier', y='Log Loss', secondary_y=True, color='g', ax=ax,
	                  label='Log Loss', lw=3, markersize=10, fontsize=10)
	ticks = [item.get_text() for item in ax.get_xticklabels()]
	# g2 = df.plot.line(marker='o', linestyle='-', x='Classifier', y='F1 Score', color='#757575', ax=ax, label='F1 Score', lw=3, markersize=10, fontsize=10)
	lines, labels = g1.get_legend_handles_labels()
	# del lines[-1]
	# del labels[-1]
	lines2, labels2 = g2.get_legend_handles_labels()
	lines3, labels3 = g3.get_legend_handles_labels()
	g1.set_ylim(0.5, 1.1)
	g2.set_ylabel('ROC AUC   &   F1 Score', fontsize=10, )
	g3.set_ylabel('Log Loss', fontsize=10)
	ax.set_xlabel('')
	ax.set_xticklabels(competitors)
	title_name = output.split('/')[0]
	df_name = "_".join(title_name.split('_')[:-1])
	maf = title_name.split('_')[-1]
	if maf == "0":
		ax.set_title('{}'.format(title_name), y=1.0, fontsize=14)
	else:
		ax.set_title('{}'.format(title_name), y=1.0, fontsize=14)

	# red_patch = mpatches.Patch(color='#757575', label='Foo')
	# lines[1] = red_patch
	# ax.legend(lines + lines3, labels + labels3, frameon=False, loc='upper right', ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=10)
	ax.legend(lines + lines3, labels + labels3, frameon=False, loc='upper center', ncol=3,
	          bbox_to_anchor=(0.5, 1.0), fontsize=10)
	ax.yaxis.grid()
	# ax.tick_params(axis='x', which='major', labelsize=12)
	plt.tight_layout()
	plt.savefig(output + '/histo_auc_f1_ll.png', dpi=600)
	plt.close()



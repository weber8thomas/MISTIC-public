import collections
import multiprocessing
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parmap
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, percentileofscore

from src.utils.utils import mkdir


def mp_exome(file, pred_dict, df_l, check):
	causative_variant = file.split('-')[-2]
	filename = file
	exome_name = file.split('-')[0].replace('PREDICTION_', '')
	source = file.split('-')[-1].replace('.csv.gz', '')

	results_file = pd.read_csv(file, compression='gzip', sep='\t', low_memory=False)
	results_file['ID'] = results_file['ID'].str.lstrip('chr_')

	for col in results_file:

		if col in check:
			tmp_d = dict()
			results_file[col].replace({0: 0, 1: 1, -1: 0}, inplace=True)
			benign_nb = results_file[results_file[col] == 0].shape[0]
			path_nb = results_file[results_file[col] == 1].shape[0]
			score_col = results_file[['ID', col, pred_dict[col]]]
			vus = score_col[pred_dict[col]].isna().sum()
			score_col_path = score_col.sort_values(by=pred_dict[col], ascending=False)['ID'].values.tolist()

			if causative_variant in score_col_path:
				index = score_col_path.index(causative_variant) + 1
				score_causative = score_col.loc[score_col['ID'] == causative_variant, pred_dict[col]].values[0]
				percentile = percentileofscore(score_col[pred_dict[col]].dropna().values, score_causative)
				tmp_d['Prediction'] = results_file[results_file['ID'] == causative_variant][col].values[0]
				tmp_d['Causative_variant'] = causative_variant
				tmp_d['Score'] = results_file[results_file['ID'] == causative_variant][pred_dict[col]].values[0]
				tmp_d['Percentile index'] = percentile

			else:
				index = len(score_col.values.tolist()) + 1

			tmp_d['Classifier'] = col
			tmp_d['Benign detected'] = benign_nb
			tmp_d['No score'] = vus
			tmp_d['Pathogenic detected'] = path_nb
			tmp_d['Ratio_path_percentage'] = path_nb / (path_nb + benign_nb)
			tmp_d['Causative index'] = index
			tmp_d['Filename'] = filename
			tmp_d['Exome_name'] = exome_name
			tmp_d['Source'] = source

			df_l.append(tmp_d)


def stats_exomes_1000G(directory):
	pred_dict = {
		"ClinPred_score_pred": "ClinPred_flag",
		"PrimateAI_score_pred": "PrimateAI_flag",
		"M-CAP_score_pred": "M-CAP_flag",
		"REVEL_score_pred": "REVEL_flag",
		"Eigen-raw_coding_pred": "Eigen-raw_coding_flag",
		"fathmm-XF_coding_score_pred": "fathmm-XF_coding_flag",
		"MISTIC_pred": "MISTIC_proba",
	}

	check = ['Eigen-raw_coding_pred', 'hsEigen-raw_coding_pred', 'M-CAP_score_pred', 'hsM-CAP_score_pred',
	         'PrimateAI_score_pred', 'hsPrimateAI_score_pred', 'ClinPred_score_pred', 'hsClinPred_score_pred',
	         'REVEL_score_pred', 'hsREVEL_score_pred', 'fathmm-XF_coding_score_pred',
	         'hsfathmm-XF_coding_score_pred', 'MISTIC_pred', 'hsMISTIC_pred']

	mkdir(directory + '/PLOTS_AND_TABLES')

	m = multiprocessing.Manager()
	df_l = m.list()

	l_dir = list(sorted(os.listdir(directory)))
	directory = directory + '/' if directory.endswith('/') is False else directory
	l_dir = [directory + f for f in l_dir if 'PLOTS' not in f]
	parmap.starmap(mp_exome, list(zip(l_dir)), pred_dict, df_l, check, pm_pbar=True)
	df_l = list(df_l)


	sorter = ['Eigen', 'PrimateAI', 'FATHMM-XF', 'ClinPred', 'REVEL', 'M-CAP', 'MISTIC']
	df_stats = pd.DataFrame(df_l).sort_values(by=['Classifier'])
	df_stats.Classifier = df_stats.Classifier.str.split('_')
	df_stats.Classifier = df_stats.Classifier.str[0]
	df_stats.Classifier = df_stats.Classifier.str.replace('-raw', '')
	df_stats.Classifier = df_stats.Classifier.str.replace('fathmm-XF', 'FATHMM-XF')


	hs_df = df_stats.copy()
	hs_df.Classifier = hs_df.Classifier.astype('category')
	hs_df.Classifier.cat.set_categories(sorter, inplace=True)



	hs_df.loc[hs_df['Percentile index'].isna() == True, 'Causative index'] = np.nan


	for ylim, name in zip([(0,50),(-5,350)], ['zoom', 'full_scale']):

		f = plt.figure(figsize=(6, 4))
		sns.violinplot(x="Classifier", y="Causative index", data=hs_df,   palette=["#0c2461", "#22a6b3", "#9b59b6", "#2196F3", "#4CAF50", "#FF9800", "#f44336"], showfliers=False, cut=0.1, linewidth=2, inner="box", scale='width')
		plt.xticks(rotation=45)

		plt.xlabel('')
		plt.ylabel('Ranking of causative deleterious variants')
		plt.ylim(ylim[0], ylim[1])
		f.tight_layout()
		plt.savefig(directory + '/PLOTS_AND_TABLES/Ranking_exomes_index_{}.png'.format(name), dpi=300)
		plt.close()

	for ylim, name in zip([(80,100),(0,100)], ['zoom', 'full_scale']):

		f = plt.figure(figsize=(8, 6))
		sns.boxplot(x="Classifier", y="Percentile index", data=hs_df,   palette=["#bdc3c7"], showfliers=False,)
		sns.stripplot(x="Classifier", y="Percentile index", data=hs_df,  palette=["#0c2461", "#22a6b3", "#9b59b6", "#2196F3", "#4CAF50", "#FF9800", "#f44336"], alpha=0.25, linewidth=0.4)
		f.tight_layout()
		plt.xlabel('')
		plt.ylabel('Percentile rank of causative variant')
		plt.ylim(ylim[0], ylim[1])
		plt.savefig(directory + '/PLOTS_AND_TABLES/Ranking_exomes_percentile_{}.png'.format(name), dpi=300)
		plt.close()

	sns.set_context("paper", font_scale=1)


	plt.figure(figsize=(6, 4))
	hs_df['Path_amount'] = 100 * (hs_df['Pathogenic detected']) / (
				hs_df['Pathogenic detected'] + hs_df['Benign detected'])
	plt.grid(axis='y', alpha=0.2)

	sns.violinplot(x="Classifier", y="Ratio_path_percentage", data=hs_df, palette=["#0c2461", "#22a6b3", "#9b59b6", "#2196F3", "#4CAF50", "#FF9800", "#f44336"], showfliers=False, inner="box", scale='width', linewidth=2)
	plt.xticks(rotation=45)

	plt.ylabel('Percentage of predicted deleterious variants')
	plt.xlabel('')

	l_hs = list()
	for clf in list(hs_df['Classifier'].unique()):
		d_hs = dict()
		values_path_amount = hs_df.loc[hs_df['Classifier'] == clf]['Ratio_path_percentage'].values
		d_hs['Classifier'] = clf
		d_hs['Path_amount_mean'] = np.mean(values_path_amount)
		d_hs['Path_amount_median'] = np.median(values_path_amount)
		d_hs['Path_amount_std'] = np.std(values_path_amount)
		values_causative_index = hs_df.loc[hs_df['Classifier'] == clf]['Causative index'].values
		d_hs['Causative_index_mean'] = np.mean(values_causative_index)
		d_hs['Causative_index_median'] = np.median(values_causative_index)
		d_hs['Causative_index_std'] = np.std(values_causative_index)
		l_hs.append(d_hs)

	final_results_mean_df = pd.DataFrame(l_hs)
	final_results_mean_df.Classifier = final_results_mean_df.Classifier.astype('category')
	final_results_mean_df.Classifier.cat.set_categories(sorter, inplace=True)
	final_results_mean_df.sort_values(by='Classifier').T.to_excel(directory + '/PLOTS_AND_TABLES/Results_mean_median_std.xlsx')


	plt.tight_layout()
	plt.savefig(directory + '/PLOTS_AND_TABLES/Pathogenic_number_test.png', dpi=600)
	plt.close()

	df_stats = df_stats[['Classifier', 'Filename', 'Exome_name', 'Source', 'Causative_variant', 'Benign detected', 'No score',
	                     'Pathogenic detected', 'Ratio_path_percentage', 'Causative index', 'Percentile index', 'Score', 'Prediction']]

	df_stats.to_csv(directory + '/PLOTS_AND_TABLES/Stats_exomes_raw.csv', sep='\t')

	cols = ['Benign detected', 'Pathogenic detected', 'Ratio_path_percentage', 'No score', 'Causative index', 'Percentile index', 'Score', 'Prediction']
	merge_df = pd.DataFrame()
	for col in cols:
		df_stats_predictors = df_stats[['Classifier', 'Filename', 'Exome_name', 'Source', col]].pivot(
			index='Exome_name', columns='Classifier', values=col).add_suffix('_' + col.replace(' ', '_'))
		df_stats_predictors.index.name = None
		df_stats_predictors.columns.name = None
		merge_df = pd.concat([merge_df, df_stats_predictors], axis=1)
	df_stats = df_stats[['Filename', 'Exome_name', 'Source', 'Causative_variant']].drop_duplicates(keep='first')
	df_stats.set_index('Exome_name', inplace=True, )
	concat_df = pd.concat([df_stats, merge_df], axis=1, sort=True)
	concat_df = concat_df[concat_df.columns.drop(list(concat_df.filter(regex='^hs')))]
	concat_df = concat_df[concat_df.columns.drop(list(concat_df.filter(regex='_detected|_Prediction|Source|No_score')))]


	stats_df = concat_df.copy()
	stats_df.loc['mean'] = concat_df.mean()
	stats_df.loc['25']   = concat_df.quantile(0.25)
	stats_df.loc['50']   = concat_df.quantile(0.5)
	stats_df.loc['75']   = concat_df.quantile(0.75)
	stats_df.loc['90']   = concat_df.quantile(0.9)
	stats_df.loc['95']   = concat_df.quantile(0.95)
	stats_df.loc['99']   = concat_df.quantile(0.99)
	stats_df.loc['std']  = concat_df.std()
	stats_df.to_excel(directory + '/PLOTS_AND_TABLES/Stats_exomes.xlsx', index=True)


	clfs = ['Eigen', 'PrimateAI', 'FATHMM-XF', 'ClinPred', 'REVEL', 'M-CAP', 'MISTIC', ]
	d = collections.defaultdict(dict)
	for clf in clfs:
		d[clf]['Path_detected_t_test'] = ttest_ind(hs_df[hs_df['Classifier'] == clf]['Ratio_path_percentage'].dropna().values, hs_df[hs_df['Classifier'] == 'MISTIC']['Ratio_path_percentage'].dropna().values)[1]
		d[clf]['Causative_index_t_test'] = ttest_ind(hs_df[hs_df['Classifier'] == clf]['Causative index'].dropna().values, hs_df[hs_df['Classifier'] == 'MISTIC']['Causative index'].dropna().values)[1]
		d[clf]['Path_detected_mannwhitneyu'] = mannwhitneyu(hs_df[hs_df['Classifier'] == clf]['Ratio_path_percentage'].dropna().values, hs_df[hs_df['Classifier'] == 'MISTIC']['Ratio_path_percentage'].dropna().values)[1]
		d[clf]['Causative_index_mannwhitneyu'] = mannwhitneyu(hs_df[hs_df['Classifier'] == clf]['Causative index'].dropna().values, hs_df[hs_df['Classifier'] == 'MISTIC']['Causative index'].dropna().values)[1]
		d[clf]['Path_detected_kruskal'] = kruskal(hs_df[hs_df['Classifier'] == clf]['Ratio_path_percentage'].dropna().values, hs_df[hs_df['Classifier'] == 'MISTIC']['Ratio_path_percentage'].dropna().values)[1]
		d[clf]['Causative_index_kruskal'] = kruskal(hs_df[hs_df['Classifier'] == clf]['Causative index'].dropna().values, hs_df[hs_df['Classifier'] == 'MISTIC']['Causative index'].dropna().values)[1]
	pd.DataFrame(d).to_excel(directory + '/PLOTS_AND_TABLES/Test_stats.xlsx')


if __name__ == '__main__':
	stats_exomes_1000G(sys.argv[1])
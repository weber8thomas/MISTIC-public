import matplotlib
import numpy as np

matplotlib.use('agg', warn=False)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def annotate_bars(ax):
	"""
	Args:
		ax (TYPE): Description
	"""
	for p in ax.patches:
		ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
		            ha='center', va='center', fontsize=11, color='gray', rotation=90, xytext=(0, 20),
		            textcoords='offset points')


def histo_and_metrics(folder,logger):
	metric = ['RandomForestClassifier', 'LogisticRegression']
	logger.info('Generating weights histogram ...')

	sns.set(style="white")
	sns.set_context("paper")

	rename_dict = {
		"29way_logOdds": "SiPhy29way",
		"CADD_phred": "CADD",
		"CCRS_score": "CCRS",
		"Condel": "Condel",
		"GERP++_RS": "GERP++",
		"HI_score": "Haploinsufficiency",
		"MetaLR_score": "MetaLR",
		"MetaSVM_score": "MetaSVM",
		"PolyPhenVal": "PolyPhen2",
		"SIFTval": "SIFT",
		"SiPhy_29way_logOdds": "SiPhy",
		"VEST4_score": "VEST4",
		"gnomAD_exomes_AF": "gnomAD(global MAF)",
		"gnomAD_exomes_AFR_AF": "gnomAD(AFR)",
		"gnomAD_exomes_AMR_AF": "gnomAD(AMR)",
		"gnomAD_exomes_ASJ_AF": "gnomAD(ASJ)",
		"gnomAD_exomes_EAS_AF": "gnomAD(EAS)",
		"gnomAD_exomes_FIN_AF": "gnomAD(FIN)",
		"gnomAD_exomes_NFE_AF": "gnomAD(NFE)",
		"gnomAD_exomes_SAS_AF": "gnomAD(SAS)",
		"phastCons100way_vertebrate": "phastCons(Ver)",
		"phastCons17way_primate": "phastCons(Pri)",
		"phastCons30way_mammalian": "phastCons(Mam)",
		"phyloP100way_vertebrate": "phyloP(Ver)",
		"phyloP17way_primate": "phyloP(Pri)",
		"phyloP30way_mammalian": "phyloP(Mam)",
	}

	cols = ['Features', 'RandomForestClassifier_VC', 'LogisticRegression_VC', 'RandomForestClassifier',
	        'LogisticRegression']
	df = pd.read_csv(folder + '/TEST/Feature_importances.csv', sep='\t', names=['Features'] + metric,
	                 header=None).dropna()
	df = df.reset_index(level=[0, 1])
	df.rename(columns=df.iloc[0], inplace=True)
	df = df[1:]  # take the data less the header row
	df.columns = cols
	df_vc = df[cols[:3]]

	df_vc['RandomForestClassifier_VC'] = pd.to_numeric(df_vc['RandomForestClassifier_VC'].str.rstrip('%'))
	df_vc['LogisticRegression_VC'] = pd.to_numeric(df_vc['LogisticRegression_VC'].str.rstrip('%'))
	check_value = 1
	df_vc['RandomForestClassifier_VC'] = (100. * df_vc['RandomForestClassifier_VC'] / df_vc['RandomForestClassifier_VC'].sum()).round(2)
	df_vc['LogisticRegression_VC'] = np.abs(df_vc['LogisticRegression_VC'])
	df_vc['LogisticRegression_VC'] = (100. * df_vc['LogisticRegression_VC'] / df_vc['LogisticRegression_VC'].sum()).round(2)
	df = df_vc[(df_vc['LogisticRegression_VC'] <= -check_value) | (df_vc['LogisticRegression_VC'] >= check_value) | (
			df_vc['RandomForestClassifier_VC'] <= -check_value) | (
			           df_vc['RandomForestClassifier_VC'] >= check_value)]
	# df['RandomForestClassifier_VC'] = df['RandomForestClassifier_VC']/100
	# df['LogisticRegression_VC'] = df['LogisticRegression_VC']/100
	df = df.sort_values(by=['LogisticRegression_VC', 'RandomForestClassifier_VC'])
	df.columns = ['Features', 'Random Forest', 'Logistic Regression']

	l = list()
	df.set_index('Features', inplace=True)
	df.rename(index=rename_dict, inplace=True)
	for col in list(df.index):
		if col not in rename_dict.values():
			rename_dict[col] = 'AAIndex_' + col
			l.append(col)
	df.rename(index=rename_dict, inplace=True)

	logger.info(df)
	ax = df.plot(kind='bar', legend=True, color=['#0D47A1', '#F57F17'])

	ax.set_ylabel('Relative weight (%)')
	# ax.set_ylim(ymin=-10)
	# fig = ax.get_figure()
	# ax = fig.get_axes()
	# ax[1].set_ylim(ymin=-10)
	plt.legend(loc='upper center')
	plt.tight_layout()
	plt.savefig(folder + '/TEST/Weights_seaborn_histo.png', dpi=600)
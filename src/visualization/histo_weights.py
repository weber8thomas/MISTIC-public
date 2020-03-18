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


def histo_and_metrics(folder,

                      ):
	metric = ['RandomForestClassifier', 'LogisticRegression']
	print('\nGenerating weights histogram ...\n')

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

	print(df)
	ax = df.plot(kind='bar', legend=True, color=['#0D47A1', '#F57F17'])

	ax.set_ylabel('Relative weight (%)')
	# ax.set_ylim(ymin=-10)
	# fig = ax.get_figure()
	# ax = fig.get_axes()
	# ax[1].set_ylim(ymin=-10)
	plt.legend(loc='upper center')
	plt.tight_layout()
	plt.savefig(folder + '/TEST/Weights_seaborn_histo.png', dpi=600)
	return l


def heatmap(folder,):
	print('\nGenerating heatmap ...\n')

	sns.set(style="white")
	sns.set_context("paper")
	df = pd.read_csv(folder + '/TRAIN/training.csv.gz', sep="\t", compression="gzip")
	df = df[df.columns.drop(list(df.filter(regex='pred')))]
	df = df.drop(['ID', 'True_Label'], axis=1)
	# l = ['gnomAD_exomes_AF', "gnomAD_exomes_NFE_AF", "gnomAD_exomes_EAS_AF", "gnomAD_exomes_ASJ_AF",
	#      "gnomAD_exomes_FIN_AF", "gnomAD_exomes_SAS_AF", "gnomAD_exomes_AFR_AF", "gnomAD_exomes_AMR_AF",
	#      "HI_score", "phastCons30way_mammalian", "phastCons17way_primate",
	#      "phastCons100way_vertebrate", "phyloP30way_mammalian", "phyloP17way_primate", "phyloP100way_vertebrate",
	#      "29way_logOdds", "GERP++_RS", "CCRS_score", ] + ['JOND940101', 'LUTR910104', 'LUTR910106', 'OGAK980101',
	#                                                       'RISJ880101', 'RUSR970101', 'KOSJ950101', 'OVEJ920104',
	#                                                       'FEND850101', 'KOSJ950104', 'NGPC000101', 'KOSJ950105',
	#                                                       'KOSJ950106', 'KOSJ950108', 'AZAE970101', 'AZAE970102',
	#                                                       'NAOD960101', 'JOND920103', 'LUTR910101', 'LUTR910107',
	#                                                       'MOHR870101', 'PRLA000102', 'DOSZ010102', 'CSEM940101',
	#                                                       'RUSR970102', 'JOHM930101', 'OVEJ920101', 'BENS940102',
	#                                                       'QUIB020101', 'CROG050101', 'BENS940103', 'GONG920101',
	#                                                       'VOGG950101', 'MUET020101', 'MUET020102', 'KANM000101',
	#                                                       'HENS920102', 'HENS920101', 'HENS920104', 'LINK010101',
	#                                                       'KOSJ950111', 'KOSJ950109', 'KOSJ950113', 'KOSJ950115',
	#                                                       'KOSJ950114', 'KOSJ950110', 'KOSJ950112', 'KAPO950101',
	#                                                       'LUTR910102', 'DAYM780301', 'ALTS910101', 'DAYM780302',
	#                                                       'OVEJ920105', 'GEOD900101', 'MCLA720101', 'MUET010101',
	#                                                       'WEIL970101', 'QU_C930102', 'QU_C930103', 'BENS940104',
	#                                                       'MIYS930101', 'NIEK910101', 'NIEK910102', 'SIMK990103',
	#                                                       'MIRL960101', 'MIYS850103', 'MIYS960102', 'MIYS990107',
	#                                                       'SIMK990104', 'VENM980101', 'BRYS930101', 'KESO980102',
	#                                                       'MIYT790101', 'WEIL970102', 'TANS760101', 'ZHAC000106',
	#                                                       'MIYS850102', 'MIYS960101', 'ZHAC000102', 'ZHAC000105',
	#                                                       'BONM030102', 'SKOJ000101', 'SKOJ970101', 'THOP960101',
	#                                                       'TOBD000102', 'FITW660101', 'HUTJ700101', 'BONM030104',
	#                                                       'MOOG990101', 'QU_C930101', ] + ["SIFTval", "PolyPhenVal",
	#                                                                                        "VEST4_score", "Condel",
	#                                                                                        "CADD_phred", "MetaLR_score",
	#                                                                                        "MetaSVM_score", ]
	# df = df[l]
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

	df.rename(columns=rename_dict, inplace=True)
	for col in df:
		if col not in list(rename_dict.values()):
			rename_dict[col] = 'AAIndex_' + col
	df.rename(columns=rename_dict, inplace=True)

	sns.set(style="white", font_scale=0.2)
	sns.set_context("paper")
	plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(22, 22))
	mypalette = ['#5659BA', '#CE93D8', '#B2EBF2', '#EEEEEE', '#AED581', '#4EAD51', '#FFBF0D', '#CA421C']

	corr = df.corr(method="spearman")
	sns.set(font_scale=0.5)

	sns.heatmap(corr, square=True, cmap=mypalette, vmin=-0.6, vmax=1, annot_kws={"size": 5},
	            cbar_kws={"ticks": [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1], "shrink": 0.5}, )
	# clustergrid = sns.clustermap(corr, square=True, annot_kws={"size": 5}, cmap=mypalette, vmin=-0.6, vmax=1,
	#                              cbar_kws={"ticks": [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]})
	# print(clustergrid.dendrogram_col.reordered_ind)
	# reordered_columns = [list(df_original.columns)[2:][col] for col in clustergrid.dendrogram_col.reordered_ind]
	# print(reordered_columns)
	plt.tight_layout()
	plt.savefig(folder + '/TRAIN/heatmap.png', dpi=180)

import multiprocessing
import os
import sys

import pandas as pd
import parmap

from MISTIC import prediction
from src.utils import utils

list_columns = ["Amino_acids", "29way_logOdds", "ALTS910101", "AZAE970101", "AZAE970102",
                "BENS940102", "BENS940103", "BENS940104", "BONM030102", "BONM030104", "BRYS930101",
                "CADD_phred", "CCRS_score", "Condel", "CROG050101", "CSEM940101", "DAYM780301",
                "DAYM780302", "DOSZ010102", "FEND850101", "FITW660101", "GEOD900101", "GERP++_RS",
                "gnomAD_exomes_AF", "gnomAD_exomes_AFR_AF", "gnomAD_exomes_AMR_AF",
                "gnomAD_exomes_ASJ_AF", "gnomAD_exomes_EAS_AF", "gnomAD_exomes_FIN_AF",
                "gnomAD_exomes_NFE_AF", "gnomAD_exomes_SAS_AF", "GONG920101", "HENS920101",
                "HENS920102", "HENS920104", "HUTJ700101", "JOHM930101", "JOND920103", "JOND940101",
                "KANM000101", "KAPO950101", "KESO980102", "KOSJ950101", "KOSJ950104", "KOSJ950105",
                "KOSJ950106", "KOSJ950108", "KOSJ950109", "KOSJ950110", "KOSJ950111", "KOSJ950112",
                "KOSJ950113", "KOSJ950114", "KOSJ950115", "LINK010101", "LUTR910101", "LUTR910102",
                "LUTR910104", "LUTR910106", "LUTR910107", "MCLA720101", "MetaLR_score",
                "MetaSVM_score", "MIRL960101", "MIYS850102", "MIYS850103", "MIYS930101", "MIYS960101",
                "MIYS960102", "MIYS990107", "MIYT790101", "MOHR870101", "MOOG990101", "MPC_score",
                "MUET010101", "MUET020101", "MUET020102", "NAOD960101", "NGPC000101", "NIEK910101",
                "NIEK910102", "OGAK980101", "OVEJ920101", "OVEJ920104", "OVEJ920105",
                "phastCons100way_vertebrate", "phastCons17way_primate", "phastCons30way_mammalian",
                "phyloP100way_vertebrate", "phyloP17way_primate", "phyloP30way_mammalian",
                "PolyPhenVal", "PRLA000102", "QU_C930101", "QU_C930102", "QU_C930103", "QUIB020101",
                "RISJ880101", "RUSR970101", "RUSR970102", "SIFTval", "SIMK990103", "SIMK990104",
                "SKOJ000101", "SKOJ970101", "TANS760101", "THOP960101", "TOBD000102", "VENM980101",
                "VEST4_score", "VOGG950101", "WEIL970101", "WEIL970102", "ZHAC000102", "ZHAC000105",
                "ZHAC000106", ]
flag = ["REVEL_score", "ClinPred_score", "M-CAP_score", "fathmm-XF_coding_score", "Eigen-raw_coding",
	                    "PrimateAI_score", ]

m = multiprocessing.Manager()
l_df_1000G = m.list()


def mp_process_df(file, model, deleterious_df, output_dir):
	params = dict(list_columns=list_columns,
	              flag=flag,
	              input=file,
	              output_dir=os.path.dirname(file),
	              model=model,
	              wt_select=True,
	              )
	return_df = prediction(params).filter(regex='flag|pred|proba|ID|True_Label', axis=1)
	exome_name = os.path.basename(file).replace('.csv.gz', '')
	return_df['Source'] = exome_name
	deleterious_variant_to_append = deleterious_df.loc[deleterious_df['Source'] == exome_name]
	return_df = return_df.append(deleterious_variant_to_append).sort_values(by='ID').reset_index(drop=True)
	col_ordered = ['ID', 'True_Label'] + list(sorted(set(list(return_df.columns)) - set(['ID', 'True_Label', 'MISTIC_proba', 'MISTIC_pred']))) + ['MISTIC_proba', 'MISTIC_pred']
	return_df = return_df[col_ordered]
	return_df.to_csv(output_dir + exome_name + '-' + str(deleterious_variant_to_append.ID.values[0]) + '-Pathogenic.csv.gz', index=False, compression='gzip', sep='\t')


if __name__ == '__main__':
	directory = sys.argv[1]
	model_dir = sys.argv[2]
	deleterious_df_path = sys.argv[3]
	output_dir = sys.argv[4]

	deleterious_df = prediction(dict(list_columns=list_columns[1:],
	                            flag=flag,
	                            input=deleterious_df_path,
	                            output_dir=os.path.dirname(deleterious_df_path),
	                            model=model_dir,
	                            wt_select=True,
	                                 ))

	deleterious_df = deleterious_df.sample(frac=1, random_state=1)

	directory = directory + '/' if directory.endswith('/') is False else directory
	output_dir = output_dir + '/' if output_dir.endswith('/') is False else output_dir
	utils.mkdir(output_dir)

	l_dir = list(sorted(os.listdir(directory)))[:deleterious_df.shape[0]]
	exomes_id = [e.replace('.csv.gz', '') for e in l_dir]
	l_dir = [directory + file for file in l_dir if file.endswith('.csv.gz')]
	deleterious_df = deleterious_df.filter(regex='flag|pred|proba|ID|True_Label', axis=1)

	deleterious_df['Source'] = pd.Series(exomes_id)
	col_ordered = ['ID', 'True_Label'] + list(sorted(set(list(deleterious_df.columns)) - set(['ID', 'True_Label', 'MISTIC_proba', 'MISTIC_pred']))) + ['MISTIC_proba', 'MISTIC_pred']
	deleterious_df = deleterious_df[col_ordered]

	parmap.starmap(mp_process_df, list(zip(l_dir))[:100], model_dir, deleterious_df, output_dir, pm_pbar=True)

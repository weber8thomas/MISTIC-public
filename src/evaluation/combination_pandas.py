import collections
import itertools

import pandas as pd
from tqdm import tqdm

from src.features.select_columns_pandas import select_columns_pandas


def equal(tmp):
	"""

	Args:
		tmp:

	Returns:

	"""
	complete_data_path = tmp.loc[tmp['True_Label'] == 1]
	complete_data_path = complete_data_path.sample(frac=1)
	complete_data_begn = tmp.loc[tmp['True_Label'] == -1]
	complete_data_begn = complete_data_begn.sample(frac=1)
	max_size = max(complete_data_path.shape[0], complete_data_begn.shape[0])
	min_size = min(complete_data_path.shape[0], complete_data_begn.shape[0])
	prop = 0.5
	t = float(round(prop / (1 - prop), 2))
	if max_size > (t * min_size):
		pass
	elif max_size < (t * min_size):
		min_size = max_size / t

	if min_size == complete_data_path.shape[0]:
		path = complete_data_path
		begn = complete_data_begn.head(n=min_size)
	if min_size == complete_data_begn.shape[0]:
		path = complete_data_path.head(n=min_size)
		begn = complete_data_begn

	complete = pd.concat([path, begn]).drop_duplicates(keep='first')
	return complete


def merging(df1, df2, logger):
	"""

	Args:
		df1:
		df2:
		logger:

	Returns:

	"""
	final_df = pd.concat([df1, df2], axis=0)
	col_ordered = ['ID', 'True_Label'] + list(sorted(set(list(final_df.columns)) - {'ID', 'True_Label'}))

	final_df = final_df[col_ordered]

	path = final_df[final_df['True_Label'] == 1]['ID']
	path = set(path.values.tolist())
	benign = final_df[final_df['True_Label'] == -1]['ID']
	benign = set(benign.values.tolist())
	inters = list(path.intersection(benign))
	logger.info(inters)
	logger.info('Intersection between pathogenic and benign : {}'.format(len(inters)))
	final_df = final_df[~final_df['ID'].isin(inters)].sort_values(by='ID').reset_index(drop=True)

	path = final_df[final_df['True_Label'] == 1]
	len_path = path.shape[0]

	benign = final_df[final_df['True_Label'] == -1]
	len_benign = benign.shape[0]

	path = path.drop_duplicates(subset='ID', keep='first')
	len_path_dupl = path.shape[0]

	benign = benign.drop_duplicates(subset='ID', keep='first')
	len_benign_dupl = benign.shape[0]

	logger.info('Common pathogenic : {}'.format(str(len_path - len_path_dupl)))
	logger.info('Common benign : {}'.format(str(len_benign - len_benign_dupl)))

	final_df = pd.concat([path, benign], axis=0)
	# print(final_df.isna().sum())
	# final_df = final_df.dropna()
	# print(final_df)
	logger.info('Pathogenic number : ' + str(len(final_df.loc[final_df['True_Label'] == 1])))
	logger.info('Begnin number : ' + str(len(final_df.loc[final_df['True_Label'] == -1])))
	return final_df


def combine_core(d, combi, CV, combi_name, l_columns, output_dir, maf_list_process, logger):
	"""

	Args:
		d:
		combi:
		CV:
		combi_name:
		l_columns:
		output_dir:
		maf_list_process:
		logger:

	Returns:

	"""
	combine_df = merging(d[combi[0]], d[combi[1]], logger)
	combine_df.drop('Source_flag', axis=1, inplace=True)
	combine_df.dropna(inplace=True)
	logger.info(combi)
	for j in tqdm(range(CV)):
		j += 1
		new_name = 'CV' + str(j) + '_all.maf_' + combi_name

		combine_df_output = equal(combine_df)
		combine_df_output.drop(['gnomAD_exomes_AC'], axis=1, inplace=True)
		if 'gnomAD_exomes_AF' not in l_columns:
			combine_df_output.drop(['gnomAD_exomes_AF'], axis=1, inplace=True)
		combine_df_output.to_csv(output_dir + new_name, compression='gzip', sep='\t', index=False)

		tmp_df_b = combine_df[combine_df['True_Label'] == -1]
		tmp_df_p = combine_df[combine_df['True_Label'] == 1]
		for maf in maf_list_process:
			new_name = 'CV' + str(j) + '_' + str(maf) + '_' + combi_name
			tmp_df_b['gnomAD_exomes_AF'] = pd.to_numeric(tmp_df_b['gnomAD_exomes_AF'])
			tmp_d_maf = tmp_df_b[tmp_df_b['gnomAD_exomes_AF'] <= maf]
			output_df = pd.concat([tmp_df_p, tmp_d_maf], axis=0)
			output_df.dropna(inplace=True)
			output_df = equal(output_df)
			output_df.drop('gnomAD_exomes_AC', axis=1, inplace=True)
			if 'gnomAD_exomes_AF' not in l_columns:
				output_df.drop(['gnomAD_exomes_AF'], axis=1, inplace=True)
			output_df.to_csv(output_dir + new_name, compression='gzip', sep='\t', index=False)
		new_name = 'CV' + str(j) + '_AC1_' + combi_name
		tmp_df_b['gnomAD_exomes_AC'] = pd.to_numeric(tmp_df_b['gnomAD_exomes_AC'])
		tmp_df_b_AC1 = tmp_df_b[tmp_df_b['gnomAD_exomes_AC'] == 1]
		output_df = pd.concat([tmp_df_p, tmp_df_b_AC1], axis=0)
		output_df.dropna(inplace=True)
		output_df = equal(output_df)
		output_df.drop('gnomAD_exomes_AC', axis=1, inplace=True)
		if 'gnomAD_exomes_AF' not in l_columns:
			output_df.drop(['gnomAD_exomes_AF'], axis=1, inplace=True)
		output_df.to_csv(output_dir + new_name, compression='gzip', sep='\t', index=False)

def combination_pandas(full_eval_set, training_set, output_dir, logger, l_columns, flag, CV=10):
	"""

	Args:
		full_eval_set:
		training_set:
		output_dir:
		logger:
		l_columns:
		flag:
		CV:

	Returns:

	"""
	output_dir += '/'
	full_df = pd.read_csv(filepath_or_buffer=full_eval_set, sep='\t', compression='gzip', encoding='utf-8',
	                      low_memory=False)
	training = \
		pd.read_csv(filepath_or_buffer=training_set, sep='\t', compression='gzip', encoding='utf-8', low_memory=False)[
			'ID'].values
	sources = list(full_df['Source_flag'].unique())
	sources_path = list(full_df[full_df['True_Label'] == 1]['Source_flag'].unique())
	sources_benign = list(full_df[full_df['True_Label'] == -1]['Source_flag'].unique())
	combinations = list(itertools.product(sources_path, sources_benign))

	l_columns_modif = l_columns + ['Source_flag', 'gnomAD_exomes_AC']
	if 'gnomAD_exomes_AF' not in l_columns:
		l_columns_modif += ['gnomAD_exomes_AF']

	full_df = select_columns_pandas(full_df, l_columns_modif, flag)

	d = collections.defaultdict()
	for s in sources:
		tmp_df = full_df[full_df['Source_flag'] == s]
		before_shape = tmp_df.shape[0]
		tmp_df = tmp_df[~tmp_df['ID'].isin(training)]
		after_shape = tmp_df.shape[0]
		logger.info('Remove {} common variants between {} and Training file'.format(str(before_shape - after_shape), s))
		d[s] = tmp_df

	for combi in combinations:
		combi_name = 'pandas_' + '|'.join(combi) + '.csv.gz'
		if 'gnom' in combi_name:
			maf_list_process = [0.01, 0.005, 0.001, 0.0001]

			combine_core(d, combi, CV, combi_name, l_columns, output_dir, maf_list_process, logger)

		if 'gnom' not in combi_name:
			logger.info(combi)

			maf_list_process = [0]

			combine_core(d, combi, CV, combi_name, l_columns, output_dir, maf_list_process, logger)


# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
import sys
from cyvcf2 import VCF, Writer
from tqdm import tqdm
import collections
import multiprocessing
import time
import pathlib


def mkdir(directory):
	"""

    Args:
        directory: str
            Directory to create

    Returns:

    """
	if not os.path.exists(directory):
		try:
			pathlib.Path(directory).mkdir(exist_ok=True)
		except FileNotFoundError:
			sys.exit("============\nSee you soon :)\n============")


def prepare_clinvar(vcf_clinvar, index_dict, vep_field, vep_separator, output_clinvar, intersection_circularity):
	o = Writer(output_clinvar, vcf_clinvar)
	stats_dict = collections.defaultdict()
	db = 'ClinVar'
	stats_dict['Database'] = db
	stats_dict['Total_SNV'] = 0
	stats_dict['Missense'] = 0
	stats_dict['Pathogenic'] = 0
	stats_dict['High_confidence'] = 0
	stats_dict['Circularity_filtering'] = 0

	for counter, variant in enumerate(tqdm(vcf_clinvar, desc=db)):
		if counter == 10000:
			break
		if len(variant.REF) == 1 and len(variant.ALT[0]) == 1 and vep_field:
			stats_dict['Total_SNV'] += 1
			id_var = str(variant.CHROM) + '_' + str(variant.POS) + '_' + str(variant.REF) + '_' + str(variant.ALT[0])
			csq = variant.INFO.get(vep_field)
			if ',' in csq:
				csq = csq.split(',')
			else:
				csq = [csq]
			l_impact = [case.split(vep_separator)[index_dict['IMPACT']] for case in csq]
			if 'HIGH' in l_impact:
				continue
			else:
				c = 0
				for case in csq:
					case = case.split(vep_separator)
					if 'missense_variant' in case[index_dict['Consequence']]:
						if c == 0:
							c += 1
							stats_dict['Missense'] += 1

							# CLNSIG
							cln_sig = variant.INFO.get('CLNSIG')
							for state in [
								'Pathogenic']:  # POSSIBILITY TO GET BENIGN VALIDATED VARIANTS BY ADDING 'Benign' IN THE LIST
								if cln_sig and state[1:] in cln_sig:
									stats_dict['Pathogenic'] += 1

									# CLNREVSTAT
									tmp_stats = variant.INFO['CLNREVSTAT'].split(',')
									good_criteria = ['criteria_provided', 'reviewed_by_expert_panel',
									                 '_multiple_submitters', 'practice_guideline']
									bad_criteria = '_conflicting_interpretations'
									if set(good_criteria) & set(tmp_stats) and bad_criteria not in tmp_stats:

										stats_dict['High_confidence'] += 1

										if id_var not in intersection_circularity:
											stats_dict['Circularity_filtering'] += 1

											variant.INFO['True_Label'] = 1
											variant.INFO['Source'] = db
											o.write_record(variant)
	return stats_dict


def prepare_hgmd(vcf_hgmd, index_dict, vep_field, vep_separator, output_hgmd, intersection_circularity):
	o = Writer(output_hgmd, vcf_hgmd)
	stats_dict = collections.defaultdict()
	db = 'HGMD'
	stats_dict['Database'] = db
	stats_dict['Total_SNV'] = 0
	stats_dict['Missense'] = 0
	stats_dict['Pathogenic'] = 0
	stats_dict['Circularity_filtering'] = 0

	for counter, variant in enumerate(tqdm(vcf_hgmd, desc=db)):
		if counter == 10000:
			break
		if len(variant.REF) == 1 and len(variant.ALT[0]) == 1 and vep_field:
			stats_dict['Total_SNV'] += 1
			id_var = str(variant.CHROM) + '_' + str(variant.POS) + '_' + str(variant.REF) + '_' + str(variant.ALT[0])
			csq = variant.INFO.get(vep_field)
			if ',' in csq:
				csq = csq.split(',')
			else:
				csq = [csq]
			l_impact = [case.split(vep_separator)[index_dict['IMPACT']] for case in csq]
			if 'HIGH' in l_impact:
				continue
			else:
				c = 0
				for case in csq:
					case = case.split(vep_separator)
					if 'missense_variant' in case[index_dict['Consequence']]:
						stats_dict['Missense'] += 1
						if c == 0:
							c += 1
							cln_sig = variant.INFO.get('CLASS')
							if cln_sig and cln_sig == 'DM':
								stats_dict['Pathogenic'] += 1

								if id_var not in intersection_circularity:
									stats_dict['Circularity_filtering'] += 1

									variant.INFO['True_Label'] = 1
									variant.INFO['Source'] = db
									o.write_record(variant)
	return stats_dict


def prepare_gnomad(vcf_gnomad, index_dict, vep_field, vep_separator, output_gnomad):
	o = Writer(output_gnomad, vcf_gnomad)
	stats_dict = collections.defaultdict()
	db = 'gnomad'
	stats_dict['Database'] = db
	stats_dict['Total_SNV'] = 0
	stats_dict['Missense'] = 0
	stats_dict['Gene_count'] = 0

	for counter, variant in enumerate(tqdm(vcf_gnomad, desc=db)):
		if counter == 5000:
			break
		if len(variant.REF) == 1 and len(variant.ALT[0]) == 1 and vep_field:

			if int(variant.INFO.get('AC')) > 0:
				stats_dict['Total_SNV'] += 1

				csq = variant.INFO.get(vep_field)
				if ',' in csq:
					csq = csq.split(',')
				else:
					csq = [csq]
				l_impact = [case.split(vep_separator)[index_dict['IMPACT']] for case in csq]
				if 'HIGH' in l_impact:
					continue

				else:
					c = 0
					for case in csq:
						case = case.split(vep_separator)

						if 'missense_variant' in case[index_dict['Consequence']]:
							if c == 0:
								stats_dict['Missense'] += 1
								c += 1

								if int(variant.INFO.get('DP')) > 30:
									variant.INFO['True_Label'] = -1
									variant.INFO['Source'] = db
									o.write_record(variant)

	return stats_dict


def prepare_benign_training_sets(vcf, output, intersection_clinvar_hgmd, intersection_circularity, db):
	o = Writer(output, vcf)
	stats_dict = collections.defaultdict()
	stats_dict['Database'] = db
	stats_dict['Circularity_filtering'] = 0
	stats_dict['High_confidence'] = 0

	for counter, variant in enumerate(
			tqdm(vcf, desc='Removing overlapping variants : [gnomAD] ∩ [ClinVar, HGMD, Training_sets]')):
		if counter == 3000:
			break
		if len(variant.REF) == 1 and len(variant.ALT[0]) == 1:

			id_var = str(variant.CHROM) + '_' + str(variant.POS) + '_' + str(variant.REF) + '_' + str(variant.ALT[0])
			if id_var not in intersection_clinvar_hgmd:
				stats_dict['High_confidence'] += 1

				if id_var not in intersection_circularity:
					stats_dict['Circularity_filtering'] += 1
					variant.INFO['True_Label'] = -1
					variant.INFO['Source'] = db
				o.write_record(variant)
	return stats_dict


def parse_header_vcf(vcf_file, vep_field=None, vep_separator=None):
	vcf = VCF(vcf_file)
	vcf.add_info_to_header(
		{'ID': 'True_Label', 'Description': 'Pathogenic/Benign labelled variant', 'Type': 'Integer', 'Number': '1'})
	vcf.add_info_to_header({'ID': 'Source', 'Description': 'File source', 'Type': 'String', 'Number': '1'})
	vcf.add_info_to_header({'ID': 'SF', 'Description': '', 'Type': 'String', 'Number': '1'})
	index_dict = dict()
	if vep_field:
		for h in vcf.header_iter():
			try:
				if h.info()['ID'] == vep_field:
					csq_header = h.info()['Description'].split(vep_separator)
					for elem in csq_header:
						index_dict[elem] = csq_header.index(elem)
			except:
				pass
	return vcf, index_dict


def check_intersection_circularity(vcf_file, set_circularity, name1, name2):
	vcf = VCF(vcf_file)
	list_variant_vcf = list()
	for counter, variant in enumerate(tqdm(vcf, desc='Checking intersection between {} and {}'.format(name1, name2))):
		if counter == 10000:
			break
		id_var = str(variant.CHROM) + '_' + str(variant.POS) + '_' + str(variant.REF) + '_' + str(variant.ALT[0])
		list_variant_vcf.append(id_var)
	list_variant_vcf = set(list_variant_vcf)
	intersection_circularity = set()
	if list_variant_vcf.intersection(set_circularity):
		intersection_circularity = list_variant_vcf.intersection(set_circularity)
	return list(intersection_circularity)


def circularity_build_list(dir_query):
	print(dir_query)
	print(os.getcwd())
	list_dir_query = os.listdir(dir_query)
	pbar = tqdm(list_dir_query)
	tmp_list_q = list()
	for fq in pbar:
		if fq.endswith('.gz') and 'CADD' not in fq:
			# if fq.endswith('.gz'):
			pbar.set_description('Building list of variants - Processing file : {}'.format(fq))
			vcf_query = VCF(dir_query + '/' + fq)
			for counter, record in enumerate(vcf_query):
				if counter == 10000:
					break
				tmp_rec = str(record.CHROM) + '_' + str(record.POS) + '_' + str(record.REF) + '_' + str(record.ALT[0])
				tmp_list_q.append(tmp_rec)
	tmp_list_q = set(tmp_list_q)
	return tmp_list_q


if __name__ == '__main__':

	start_time = time.time()

	# HANDLE HGMD INPUTS
	hgmd_file = sys.argv[1]
	hgmd_vep_field = sys.argv[2]
	hgmd_vep_separator = sys.argv[3]


	clinvar_file = "data/raw/deleterious/clinvar_20180930_annot.vcf.gz"
	gnomad_file = "data/raw/population/gnomad.exomes.r2.1.1.sites.vcf.bgz"

	deleterious_files = "data/raw/deleterious"
	var_list_circularity_clinvar_hgmd = circularity_build_list(deleterious_files)

	training_sets_directory = "data/raw/training_sets"
	var_list_circularity = circularity_build_list(training_sets_directory)

	with ThreadPoolExecutor(2) as executor:
		# these return immediately and are executed in parallel, on separate processes
		clinvar_intersection = executor.submit(lambda p: check_intersection_circularity(*p),
		                                       [clinvar_file, var_list_circularity, 'ClinVar',
		                                        'Features training sets data'], )
		hgmd_intersection = executor.submit(lambda p: check_intersection_circularity(*p),
		                                    [hgmd_file, var_list_circularity, 'HGMD', 'Features training sets data'])

	clinvar_intersection = clinvar_intersection.result()
	hgmd_intersection = hgmd_intersection.result()

	output_clinvar = "data/processed" + '/clinvar_clean_data.vcf.gz'
	output_hgmd = "data/processed" + '/hgmd_clean_data.vcf.gz'
	output_gnomad_tmp = "data/interim" + '/gnomad_tmp_data.vcf.gz'
	output_gnomad_final = "data/processed" + '/gnomad_clean_data.vcf.gz'

	m = multiprocessing.Manager()
	stats_dict = m.dict()

	vcf_clinvar, index_dict_clinvar = parse_header_vcf(clinvar_file, 'CSQ', '|')
	arr1 = [vcf_clinvar, index_dict_clinvar, 'CSQ', '|', output_clinvar, clinvar_intersection]

	# IF VEP ANNOTATION WITH DIFFERENT NAME AND SEPARATOR, CHANGE THEM HERE
	vcf_hgmd, index_dict_hgmd = parse_header_vcf(hgmd_file, hgmd_vep_field, hgmd_vep_separator)
	arr2 = [vcf_hgmd, index_dict_hgmd, hgmd_vep_field, hgmd_vep_separator, output_hgmd, hgmd_intersection]

	with ThreadPoolExecutor(2) as executor:
		future_1 = executor.submit(lambda p: prepare_clinvar(*p), arr1)
		future_2 = executor.submit(lambda p: prepare_hgmd(*p), arr2)

	result_1 = future_1.result()
	result_2 = future_2.result()

	vcf_gnomad, index_dict_gnomad = parse_header_vcf(gnomad_file, 'vep', '|')

	result_3 = prepare_gnomad(vcf_gnomad, index_dict_gnomad, 'vep', '|', output_gnomad_tmp)

	with ThreadPoolExecutor(2) as executor:
		gnomad_intersection_training_sets = executor.submit(lambda p: check_intersection_circularity(*p),
		                                                    [output_gnomad_tmp, var_list_circularity, 'gnomAD',
		                                                     'Training sets'])
		gnomad_intersection_clinvar_hgmd = executor.submit(lambda p: check_intersection_circularity(*p),
		                                                   [output_gnomad_tmp, var_list_circularity_clinvar_hgmd,
		                                                    'gnomAD', 'ClinVar HGMD'])

	gnomad_intersection_training_sets = gnomad_intersection_training_sets.result()
	gnomad_intersection_clinvar_hgmd = gnomad_intersection_clinvar_hgmd.result()

	vcf_gnomad_tmp, index_dict_gnomad = parse_header_vcf(output_gnomad_tmp)

	result_4 = prepare_benign_training_sets(vcf_gnomad_tmp, output_gnomad_final, gnomad_intersection_clinvar_hgmd,
	                                        gnomad_intersection_training_sets, 'gnomad')

	if os.path.isfile(output_gnomad_tmp):
		os.remove(output_gnomad_tmp)

	result_3['Circularity_filtering'] = result_4['Circularity_filtering']
	result_3['High_confidence'] = result_4['High_confidence']

	data_df = [result_1, result_2, result_3]
	df_stats = pd.DataFrame(data_df)
	df_stats = df_stats[
		['Database', 'Total_SNV', 'Missense', 'Pathogenic', 'High_confidence', 'Circularity_filtering']].set_index(
		'Database')
	df = df_stats.T
	print(df)
	df.to_csv("data/processed" + '/raw_data_stats.csv', sep='\t')

	elapsed_time = time.time() - start_time
	print(time.strftime('---- Elapsed Time - %H:%M:%S ----', time.gmtime(elapsed_time)))

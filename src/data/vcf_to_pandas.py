import argparse
from pprint import pprint
import numpy as np
import pandas as pd
from cyvcf2 import VCF
from tqdm import tqdm


class Parse:
	"""
Function build to convert VCF file to Pandas matrix

Args:
	true_label (string): label
	vcfanno (list): list
	vep (list): list
	output (string): output
	fname (string): input

Returns:
	None
"""

	def __init__(self,
	             fname,
	             output,
	             vcfanno=list(),
	             true_label=0,
	             vep=list(),
	             vep_field='CSQ',
	             vep_separator='|'):
		"""

		Args:
			fname:
			output:
			vcfanno:
			true_label:
			vep_field:
			vep_separator:
		"""

		nan_dict = {
			'': np.nan,
			'.': np.nan,
			'NaN': np.nan,
			'nan': np.nan,
			'None': np.nan,
			'NA': np.nan,
		}

		unwanted = ['', '.', 'NaN', 'nan', 'None', 'NA']

		print('\nStarting VCF to Pandas ...')

		v = VCF(fname)

		l_dict = list()
		index_dict = dict()
		if vep_field:
			for h in v.header_iter():
				print(h)
				try:
					if h.info()['ID'] == vep_field:
						csq_header = h.info()['Description'].split(vep_separator)
						for elem in csq_header:
							index_dict[elem] = csq_header.index(elem)
				except:
					pass

		pprint(csq_header)
		pprint(index_dict)

		for record in tqdm(v):
			if len(record.REF) == 1 and len(record.ALT[0]) == 1:
				tmp_dict = dict()
				id = str(record.CHROM) + '_' + str(record.POS) + '_' + str(record.REF) + '_' + str(record.ALT[0])
				tmp_dict['ID'] = id
				if vep_field:
					csq = record.INFO.get(vep_field)
					if ',' in csq:
						csq = csq.split(',')
					else:
						csq = [csq]
					for case in csq:
						case = case.split(vep_separator)
						if 'missense_variant' in case[1]:
							tmp_dict['Amino_acids'] = case[index_dict['Amino_acids']]
				if true_label:
					tmp_dict['True_Label'] = true_label
				else:
					tmp_dict['True_Label'] = record.INFO.get('True_Label')
				# vcfanno=list()
				for col in vcfanno:
					field = str(record.INFO.get(col))
					if field:
						if ',' in field:
							field = field.split(',')
							if len(set(field)) <= 1:
								field = field[0]
							else:
								try:
									field = [float(e) for e in field if e not in unwanted]
									genename = False
								except:
									field = list(set(field))[0]
									genename = True
								if 'SIFT' in col and genename is False:
									field = min(field)
								elif genename is False:
									field = max(field)
								elif genename is True:
									field = field

						tmp_dict[col] = field
				l_dict.append(tmp_dict)

		final_df = pd.DataFrame(data=l_dict)
		col_ordered = ['ID', 'True_Label'] + list(sorted(set(list(final_df.columns)) - {'ID', 'True_Label'}))
		final_df = final_df[col_ordered]
		final_df = final_df.replace(nan_dict)

		final_df.to_csv(path_or_buf=output, index=False, compression='gzip', sep='\t')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='From VCF to CSV', usage='%(prog)s [-h] [-i INI]')

	required = parser.add_argument_group('required arguments')

	required.add_argument('-f', '--file_name',
	                      metavar='',
	                      type=str,
	                      required=True,
	                      help='Name of the VCF file')

	required.add_argument('-o', '--output_name',
	                      type=str,
	                      required=True,
	                      help='Name of the output pandas file')

	parser.add_argument('--vep',
	                    metavar='',
	                    nargs="*",
	                    type=str,
	                    help='Which values of INFO.CSQ features you want to extract')

	parser.add_argument('--vcfanno',
	                    metavar='',
	                    nargs="*",
	                    type=str,
	                    help='Which values of INFO.CSQ features you want to extract')

	parser.add_argument('-l', '--label',
	                    metavar='',
	                    type=str,
	                    help='Label')

	parser.add_argument('--vep_field',
	                    metavar='',
	                    type=str,
	                    help='Label',
	                    default='CSQ')

	parser.add_argument('--vep_separator',
	                    metavar='',
	                    type=str,
	                    help='Label',
	                    default='|')

	args = parser.parse_args()
	arg_dict = vars(args)
	Parse(fname=arg_dict['file_name'],
	      output=arg_dict['output_name'],
	      vcfanno=arg_dict['vcfanno'],
	      true_label=arg_dict['label'],
	      vep_field=arg_dict['vep_field'],
	      vep_separator=arg_dict['vep_separator'],
	      )

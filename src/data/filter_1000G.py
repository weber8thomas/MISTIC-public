import collections
import pandas as pd
import os
import sys
from cyvcf2 import VCF, Writer
# import vcf
from pprint import pprint
from tqdm import tqdm
import re


def get_header(vcf, vep_field, vep_separator):
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
	return index_dict


vcf = VCF(sys.argv[1])

vcf.add_info_to_header({
	'ID': 'True_Label',
	'Description': 'True_Label of the variation',
	'Type': 'String',
	'Number': '1',
})

stats_dict = dict()
output = sys.argv[2]
o = Writer(output, vcf)
DB = str(sys.argv[3])
vep_field = sys.argv[4]
vep_separator = sys.argv[5]
# o_p = Writer(sys.argv[2], vcf)
# o_b = Writer(sys.argv[4], vcf)
# DB = str(sys.argv[4])
# other = sys.argv[4]

total = 0
snv = 0
missense = 0
impact = 0
pathogenic = 0
benign = 0
reviewed = 0
reviewed_b = 0
freq = 0
dp = 0
hom = 0
maf = 0
hom = 0
filter_training = 0
tmp_list = list()

if DB == 'missense':
	index_dict = get_header(vcf, vep_field, vep_separator)
	for record in tqdm(vcf):
		total += 1
		csq = record.INFO.get(vep_field).split(',')
		tmp_missense = 0
		cancer_check = 0

		# SNV
		if len(record.REF) == 1 and len(record.ALT) == 1 and len(record.ALT[0]) == 1:
			snv += 1
			l_impact = [case.split(vep_separator)[index_dict['IMPACT']] for case in csq]
			if 'HIGH' in l_impact:
				continue
			else:
				for case in csq:
					case = case.split('|')
					if 'missense' in case[1]:
						tmp_missense += 1

				# MISSENSE
				if tmp_missense >= 1:
					missense += 1
					o.write_record(record)



if DB == '1KG':

	for record in tqdm(vcf):
		total += 1
		if len(record.REF) == 1 and len(record.ALT) == 1 and len(record.ALT[0]) == 1:
			snv += 1
			# tmp_dp = int(record.INFO.get('DP'))
			# if tmp_dp >= 10:
			#     dp += 1
			AF = float(record.INFO.get('AF'))
			if AF <= 0.1:
				maf += 1
				o.write_record(record)




print('total   : \t', total)
print('snv     : \t', snv)
print('DP     : \t', dp)
print('missense : \t', missense)
print('MAF     : \t', maf)
print('HOM    : \t', hom)
# print('Filter Training     : \t', filter_training)


l = list()
pandas_dict = dict()
pandas_dict['Exome'] = output.replace('.vcf.gz', '')
pandas_dict['Total'] = total
pandas_dict['SNV'] = snv
pandas_dict['DP'] = dp
pandas_dict['MAF'] = maf
pandas_dict['HOMOZYGOUS'] = hom
pandas_dict['Impact'] = impact
pandas_dict['Missense'] = missense
# pandas_dict['Filter Training'] = filter_training
l.append(pandas_dict)
df = pd.DataFrame(data=l)
df = df[["Exome", "Total", "SNV", "DP", "Missense", "MAF"]]
df = df.set_index('Exome')
df = df.T
df.to_csv(output.replace('.vcf.gz', '') + '.csv', sep='\t')

# print('pathogenic : \t', pathogenic)
# print('benign : \t', benign)
# print('reviewed : \t', reviewed)
# print('freq     : \t', freq)
# print('hom     : \t', hom)
pprint(stats_dict)


import sys
from cyvcf2 import VCF, Writer
from tqdm import tqdm


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

output = sys.argv[2]
o = Writer(output, vcf)
vep_field = sys.argv[3]
vep_separator = sys.argv[4]

index_dict = get_header(vcf, vep_field, vep_separator)
for record in tqdm(vcf):
	csq = record.INFO.get(vep_field).split(',')
	tmp_missense = 0
	cancer_check = 0

	# SNV
	if len(record.REF) == 1 and len(record.ALT) == 1 and len(record.ALT[0]) == 1:
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
				o.write_record(record)

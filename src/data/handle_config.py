import os

import yaml
import argparse
from pprint import pprint

parser = argparse.ArgumentParser(
		description='Toolbox designed to build ML outputs based on Scikit-Learn. Used to build MISTIC and mMISTIC outputs',
		usage='%(prog)s [--help]')

parser.add_argument('--hgmd_file',
                    type=str,
                    required=True,
                    help='HGMD location file')

parser.add_argument('--hgmd_vep_field',
                    type=str,
                    required=False,
                    default='CSQ',
                    help='VEP annotation field for HGMD file, default=CSQ')

parser.add_argument('--hgmd_vep_separator',
                    type=str,
                    required=False,
                    default='|',
                    help='VEP annotation separator for HGMD file, default=|')

parser.add_argument('--clinvar_file',
                    type=str,
                    required=False,
                    help='Clinvar location file')

parser.add_argument('--clinvar_vep_field',
                    type=str,
                    required=False,
                    help='VEP annotation field for ClinVar file, default=CSQ')

parser.add_argument('--clinvar_vep_separator',
                    type=str,
                    required=False,
                    help='VEP annotation separator for ClinVar file , default=|')

parser.add_argument('--gnomad_file',
                    type=str,
                    required=False,
                    help='gnomAD location file')

parser.add_argument('--gnomad_vep_field',
                    type=str,
                    required=False,
                    help='VEP annotation field for gnomAD file, default=CSQ')

parser.add_argument('--gnomad_vep_separator',
                    type=str,
                    required=False,
                    help='VEP annotation separator for gnomAD file, default=|')

args = parser.parse_args()

arg_dict = vars(args)
arg_dict = {k:v for k,v in arg_dict.items() if v}

path_config_file = 'src/data/config.yml'
try:
	config = yaml.load(open(path_config_file), Loader=yaml.FullLoader)
	for k, v in arg_dict.items():
		db = k.split('_')[0]
		config[db][k] = v
	print('--HGMD config file changed--')
	pprint(config)
	yaml.dump(config, open(path_config_file, 'w'))
except:
	exit('HGMD config file error')
# coding=utf-8
import logging
import os
import pathlib
import sys
import pandas as pd
import csv
import gzip
import mimetypes
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np


def setup_custom_logger(name, output_dir):
	"""

	Args:
		name:
		output_dir:

	Returns:

	"""
	formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
	                              datefmt='%Y-%m-%d %H:%M:%S')
	mkdir('Logging')
	# handler = logging.FileHandler('Logging/' + output_dir + '_log.txt', mode='w')
	# handler.setFormatter(formatter)
	screen_handler = logging.StreamHandler(stream=sys.stdout)
	screen_handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)
	# logger.addHandler(handler)
	logger.addHandler(screen_handler)
	return logger


def write_to_html_file(df, title='', filename='out.html'):
	"""

	Args:
		df: Pandas Dataframe
			Dataframe to display with an html page
		title: str
			Title of the html page
		filename: str
			Output file name

	Returns:
		None
	"""
	result = '''
<html>
<head>
<style>

	h2 {
		text-align: center;
		font-family: Helvetica, Arial, sans-serif;
	}
	table { 
		margin-left: auto;
		margin-right: auto;
	}
	table, th, td {
		border: 1px solid black;
		border-collapse: collapse;
	}
	th, td {
		padding: 5px;
		text-align: center;
		font-family: Helvetica, Arial, sans-serif;
		font-size: 90%;
	}
	table tbody tr:hover {
		background-color: #dddddd;
	}
	.wide {
		width: 90%; 
	}

</style>
</head>
<body>
	'''
	result += '<h2> %s </h2>\n' % title
	result += df.to_html(classes='wide', escape=False, index=False)
	result += '''
</body>
</html>
'''
	with open(filename, 'w') as f:
		f.write(result)


def merge_dataframe(data1, data2):
	"""
	Method to merge two dataframe

	Args:
		data1: Pandas Dataframe
		data2: Pandas Dataframe

	Returns:
		Merged dataframe

	"""
	return pd.merge(data1, data2, right_index=True, left_index=True)


def get_data(input_data):
	"""
	Check if data is compressed, binary and find the good separator
	to return a Pandas Dataframe

	Args:
		input_data: input CSV, TSV, TAB, TXT data with columns

	Returns:
		Pandas Dataframe

	"""
	# os.chdir(os.path.dirname(os.path.realpath(input_data)))
	sniffer = csv.Sniffer()
	mime = mimetypes.guess_type(input_data)
	if mime[1] is None:
		i = open(input_data)
		h = next(i)
		i.close()
		dialect = sniffer.sniff(h)
		sep = dialect.delimiter
		data = pd.read_csv(input_data, sep=sep)
		return data

	elif mime[1] is not None:
		i = gzip.open(input_data, 'rb')
		h = i.readline().decode('utf-8')
		i.close()
		dialect = sniffer.sniff(h)
		sep = dialect.delimiter
		data = pd.read_csv(input_data, sep=sep, compression='gzip')
		return data


def prepare_input_data(input_data,
                       fill_blanks=True,
                       strategy="median",
                       standardize=True,
                       pred=False
                       ):
	"""

	Args:
		input_data:
		fill_blanks:
		strategy:
		standardize:
		pred:

	Returns:

	"""
	data = get_data(input_data=input_data)
	# print(data)
	df = data[data.columns.drop(list(data.filter(regex='pred|flag')))]
	predicted_labels = data.filter(regex='pred|flag')
	cols = df.columns
	columns_full = list(cols)
	columns_without_ids = columns_full
	columns_without_ids.remove('ID')
	columns_without_ids.remove('True_Label')
	info = data[['ID', 'True_Label']]

	data_without_blanks = pd.DataFrame()
	if fill_blanks is True:
		data_without_blanks = filling_blank_cells(df.drop(['ID', 'True_Label'], axis=1), strategy=strategy)
	if fill_blanks is False:
		data_without_blanks = df.drop(['ID', 'True_Label'], axis=1).dropna()
	if standardize is True:
		data_without_blanks = StandardScaler().fit_transform(data_without_blanks)
	if standardize is False:
		pass
	data_without_blanks.columns = columns_without_ids
	complete_data = merge_dataframe(info, pd.DataFrame(data_without_blanks))
	complete_data = complete_data.join(predicted_labels, how='left')
	if pred is False:
		complete_data = complete_data.dropna()

	# print('\n')
	return complete_data


def filling_blank_cells(data_inp, strategy):
	"""
	Method to fill blank cells into a Pandas Dataframe

	Args:
		data_inp: Pandas Dataframe
			Dataframe without ids, only numeric data
		strategy: str
			Determine the strategy to fill blank cells
			"mean", "median", "most_frequent" or "constant", check :
			http://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

	Returns:
		Dataframe without blank cells

	"""
	imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
	data_out = imp.fit(data_inp)
	data_out = data_out.transform(data_inp)
	pd_data = pd.DataFrame(data_out)
	return pd_data


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
			logging.error('Unable to find or create directory {}'.format(directory))
			sys.exit("============\nSee you soon :)\n============")

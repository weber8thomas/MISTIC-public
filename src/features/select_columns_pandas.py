import ast
from src.features import AAIndex
import numpy as np
import pandas as pd
from tqdm import tqdm

aa = AAIndex.AAIndex()


def new_columns_matrix(array):
	list_matrix = list()
	AA = list()
	for x in array:
		if '/' in x:
			AA = x.split('/')
		elif '_' in x:
			AA = x.split('_')

		ref = AA[0]
		alt = AA[1]
		ref = 'C' if ref == 'U' else ref
		alt = 'C' if alt == 'U' else alt
		score = aa.score_missense(ref, alt)
		list_matrix.append(score)
	return list_matrix


def select_columns_pandas(df, list_columns, flag, progress_bar=True, HS=False):
	list_columns = ['ID', 'True_Label'] + list_columns

	matrix_cols = ['ANDN920101', 'ARGP820101', 'ARGP820102', 'ARGP820103', 'BEGF750101', 'BEGF750102', 'BEGF750103',
	               'BHAR880101', 'BIGC670101', 'BIOV880101', 'BIOV880102', 'BROC820101', 'BROC820102', 'BULH740101',
	               'BULH740102', 'BUNA790101', 'BUNA790102', 'BUNA790103', 'BURA740101', 'BURA740102', 'CHAM810101',
	               'CHAM820101', 'CHAM820102', 'CHAM830101', 'CHAM830102', 'CHAM830103', 'CHAM830104', 'CHAM830105',
	               'CHAM830106', 'CHAM830107', 'CHAM830108', 'CHOC750101', 'CHOC760101', 'CHOC760102', 'CHOC760103',
	               'CHOC760104', 'CHOP780101', 'CHOP780201', 'CHOP780202', 'CHOP780203', 'CHOP780204', 'CHOP780205',
	               'CHOP780206', 'CHOP780207', 'CHOP780208', 'CHOP780209', 'CHOP780210', 'CHOP780211', 'CHOP780212',
	               'CHOP780213', 'CHOP780214', 'CHOP780215', 'CHOP780216', 'CIDH920101', 'CIDH920102', 'CIDH920103',
	               'CIDH920104', 'CIDH920105', 'COHE430101', 'CRAJ730101', 'CRAJ730102', 'CRAJ730103', 'DAWD720101',
	               'DAYM780101', 'DAYM780201', 'DESM900101', 'DESM900102', 'EISD840101', 'EISD860101', 'EISD860102',
	               'EISD860103', 'FASG760101', 'FASG760102', 'FASG760103', 'FASG760104', 'FASG760105', 'FAUJ830101',
	               'FAUJ880101', 'FAUJ880102', 'FAUJ880103', 'FAUJ880104', 'FAUJ880105', 'FAUJ880106', 'FAUJ880107',
	               'FAUJ880108', 'FAUJ880109', 'FAUJ880110', 'FAUJ880111', 'FAUJ880112', 'FAUJ880113', 'FINA770101',
	               'FINA910101', 'FINA910102', 'FINA910103', 'FINA910104', 'GARJ730101', 'GEIM800101', 'GEIM800102',
	               'GEIM800103', 'GEIM800104', 'GEIM800105', 'GEIM800106', 'GEIM800107', 'GEIM800108', 'GEIM800109',
	               'GEIM800110', 'GEIM800111', 'GOLD730101', 'GOLD730102', 'GRAR740101', 'GRAR740102', 'GRAR740103',
	               'GUYH850101', 'HOPA770101', 'HOPT810101', 'HUTJ700101', 'HUTJ700102', 'HUTJ700103', 'ISOY800101',
	               'ISOY800102', 'ISOY800103', 'ISOY800104', 'ISOY800105', 'ISOY800106', 'ISOY800107', 'ISOY800108',
	               'JANJ780101', 'JANJ780102', 'JANJ780103', 'JANJ790101', 'JANJ790102', 'JOND750101', 'JOND750102',
	               'JOND920101', 'JOND920102', 'JUKT750101', 'JUNJ780101', 'KANM800101', 'KANM800102', 'KANM800103',
	               'KANM800104', 'KARP850101', 'KARP850102', 'KARP850103', 'KHAG800101', 'KLEP840101', 'KRIW710101',
	               'KRIW790101', 'KRIW790102', 'KRIW790103', 'KYTJ820101', 'LAWE840101', 'LEVM760101', 'LEVM760102',
	               'LEVM760103', 'LEVM760104', 'LEVM760105', 'LEVM760106', 'LEVM760107', 'LEVM780101', 'LEVM780102',
	               'LEVM780103', 'LEVM780104', 'LEVM780105', 'LEVM780106', 'LEWP710101', 'LIFS790101', 'LIFS790102',
	               'LIFS790103', 'MANP780101', 'MAXF760101', 'MAXF760102', 'MAXF760103', 'MAXF760104', 'MAXF760105',
	               'MAXF760106', 'MCMT640101', 'MEEJ800101', 'MEEJ800102', 'MEEJ810101', 'MEEJ810102', 'MEIH800101',
	               'MEIH800102', 'MEIH800103', 'MIYS850101', 'NAGK730101', 'NAGK730102', 'NAGK730103', 'NAKH900101',
	               'NAKH900102', 'NAKH900103', 'NAKH900104', 'NAKH900105', 'NAKH900106', 'NAKH900107', 'NAKH900108',
	               'NAKH900109', 'NAKH900110', 'NAKH900111', 'NAKH900112', 'NAKH900113', 'NAKH920101', 'NAKH920102',
	               'NAKH920103', 'NAKH920104', 'NAKH920105', 'NAKH920106', 'NAKH920107', 'NAKH920108', 'NISK800101',
	               'NISK860101', 'NOZY710101', 'OOBM770101', 'OOBM770102', 'OOBM770103', 'OOBM770104', 'OOBM770105',
	               'OOBM850101', 'OOBM850102', 'OOBM850103', 'OOBM850104', 'OOBM850105', 'PALJ810101', 'PALJ810102',
	               'PALJ810103', 'PALJ810104', 'PALJ810105', 'PALJ810106', 'PALJ810107', 'PALJ810108', 'PALJ810109',
	               'PALJ810110', 'PALJ810111', 'PALJ810112', 'PALJ810113', 'PALJ810114', 'PALJ810115', 'PALJ810116',
	               'PARJ860101', 'PLIV810101', 'PONP800101', 'PONP800102', 'PONP800103', 'PONP800104', 'PONP800105',
	               'PONP800106', 'PONP800107', 'PONP800108', 'PRAM820101', 'PRAM820102', 'PRAM820103', 'PRAM900101',
	               'PRAM900102', 'PRAM900103', 'PRAM900104', 'PTIO830101', 'PTIO830102', 'QIAN880101', 'QIAN880102',
	               'QIAN880103', 'QIAN880104', 'QIAN880105', 'QIAN880106', 'QIAN880107', 'QIAN880108', 'QIAN880109',
	               'QIAN880110', 'QIAN880111', 'QIAN880112', 'QIAN880113', 'QIAN880114', 'QIAN880115', 'QIAN880116',
	               'QIAN880117', 'QIAN880118', 'QIAN880119', 'QIAN880120', 'QIAN880121', 'QIAN880122', 'QIAN880123',
	               'QIAN880124', 'QIAN880125', 'QIAN880126', 'QIAN880127', 'QIAN880128', 'QIAN880129', 'QIAN880130',
	               'QIAN880131', 'QIAN880132', 'QIAN880133', 'QIAN880134', 'QIAN880135', 'QIAN880136', 'QIAN880137',
	               'QIAN880138', 'QIAN880139', 'RACS770101', 'RACS770102', 'RACS770103', 'RACS820101', 'RACS820102',
	               'RACS820103', 'RACS820104', 'RACS820105', 'RACS820106', 'RACS820107', 'RACS820108', 'RACS820109',
	               'RACS820110', 'RACS820111', 'RACS820112', 'RACS820113', 'RACS820114', 'RADA880101', 'RADA880102',
	               'RADA880103', 'RADA880104', 'RADA880105', 'RADA880106', 'RADA880107', 'RADA880108', 'RICJ880101',
	               'RICJ880102', 'RICJ880103', 'RICJ880104', 'RICJ880105', 'RICJ880106', 'RICJ880107', 'RICJ880108',
	               'RICJ880109', 'RICJ880110', 'RICJ880111', 'RICJ880112', 'RICJ880113', 'RICJ880114', 'RICJ880115',
	               'RICJ880116', 'RICJ880117', 'ROBB760101', 'ROBB760102', 'ROBB760103', 'ROBB760104', 'ROBB760105',
	               'ROBB760106', 'ROBB760107', 'ROBB760108', 'ROBB760109', 'ROBB760110', 'ROBB760111', 'ROBB760112',
	               'ROBB760113', 'ROBB790101', 'ROSG850101', 'ROSG850102', 'ROSM880101', 'ROSM880102', 'ROSM880103',
	               'SIMZ760101', 'SNEP660101', 'SNEP660102', 'SNEP660103', 'SNEP660104', 'SUEM840101', 'SUEM840102',
	               'SWER830101', 'TANS770101', 'TANS770102', 'TANS770103', 'TANS770104', 'TANS770105', 'TANS770106',
	               'TANS770107', 'TANS770108', 'TANS770109', 'TANS770110', 'VASM830101', 'VASM830102', 'VASM830103',
	               'VELV850101', 'VENT840101', 'VHEG790101', 'WARP780101', 'WEBA780101', 'WERD780101', 'WERD780102',
	               'WERD780103', 'WERD780104', 'WOEC730101', 'WOLR810101', 'WOLS870101', 'WOLS870102', 'WOLS870103',
	               'YUTK870101', 'YUTK870102', 'YUTK870103', 'YUTK870104', 'ZASB820101', 'ZIMJ680101', 'ZIMJ680102',
	               'ZIMJ680103', 'ZIMJ680104', 'ZIMJ680105', 'AURR980101', 'AURR980102', 'AURR980103', 'AURR980104',
	               'AURR980105', 'AURR980106', 'AURR980107', 'AURR980108', 'AURR980109', 'AURR980110', 'AURR980111',
	               'AURR980112', 'AURR980113', 'AURR980114', 'AURR980115', 'AURR980116', 'AURR980117', 'AURR980118',
	               'AURR980119', 'AURR980120', 'ONEK900101', 'ONEK900102', 'VINM940101', 'VINM940102', 'VINM940103',
	               'VINM940104', 'MUNV940101', 'MUNV940102', 'MUNV940103', 'MUNV940104', 'MUNV940105', 'WIMW960101',
	               'KIMC930101', 'MONM990101', 'BLAM930101', 'PARS000101', 'PARS000102', 'KUMS000101', 'KUMS000102',
	               'KUMS000103', 'KUMS000104', 'TAKK010101', 'FODM020101', 'NADH010101', 'NADH010102', 'NADH010103',
	               'NADH010104', 'NADH010105', 'NADH010106', 'NADH010107', 'MONM990201', 'KOEP990101', 'KOEP990102',
	               'CEDJ970101', 'CEDJ970102', 'CEDJ970103', 'CEDJ970104', 'CEDJ970105', 'FUKS010101', 'FUKS010102',
	               'FUKS010103', 'FUKS010104', 'FUKS010105', 'FUKS010106', 'FUKS010107', 'FUKS010108', 'FUKS010109',
	               'FUKS010110', 'FUKS010111', 'FUKS010112', 'MITS020101', 'TSAJ990101', 'TSAJ990102', 'COSI940101',
	               'PONP930101', 'WILM950101', 'WILM950102', 'WILM950103', 'WILM950104', 'KUHL950101', 'GUOD860101',
	               'JURD980101', 'BASU050101', 'BASU050102', 'BASU050103', 'SUYM030101', 'PUNT030101', 'PUNT030102',
	               'GEOR030101', 'GEOR030102', 'GEOR030103', 'GEOR030104', 'GEOR030105', 'GEOR030106', 'GEOR030107',
	               'GEOR030108', 'GEOR030109', 'ZHOH040101', 'ZHOH040102', 'ZHOH040103', 'BAEK050101', 'HARY940101',
	               'PONJ960101', 'DIGM050101', 'WOLR790101', 'OLSK800101', 'KIDA850101', 'GUYH850102', 'GUYH850104',
	               'GUYH850105', 'JACR890101', 'COWR900101', 'BLAS910101', 'CASG920101', 'CORJ870101', 'CORJ870102',
	               'CORJ870103', 'CORJ870104', 'CORJ870105', 'CORJ870106', 'CORJ870107', 'CORJ870108', 'MIYS990101',
	               'MIYS990102', 'MIYS990103', 'MIYS990104', 'MIYS990105', 'ENGD860101', 'FASG890101', 'KARS160101',
	               'KARS160102', 'KARS160103', 'KARS160104', 'KARS160105', 'KARS160106', 'KARS160107', 'KARS160108',
	               'KARS160109', 'KARS160110', 'KARS160111', 'KARS160112', 'KARS160113', 'KARS160114', 'KARS160115',
	               'KARS160116', 'KARS160117', 'KARS160118', 'KARS160119', 'KARS160120', 'KARS160121', 'KARS160122',
	               'ALTS910101', 'BENS940101', 'BENS940102', 'BENS940103', 'BENS940104', 'CSEM940101', 'DAYM780301',
	               'FEND850101', 'FITW660101', 'GEOD900101', 'GONG920101', 'GRAR740104', 'HENS920101', 'HENS920102',
	               'HENS920103', 'JOHM930101', 'JOND920103', 'JOND940101', 'KOLA920101', 'LEVJ860101', 'LUTR910101',
	               'LUTR910102', 'LUTR910103', 'LUTR910104', 'LUTR910105', 'LUTR910106', 'LUTR910107', 'LUTR910108',
	               'LUTR910109', 'MCLA710101', 'MCLA720101', 'MIYS930101', 'MIYT790101', 'MOHR870101', 'NIEK910101',
	               'NIEK910102', 'OVEJ920101', 'QU_C930101', 'QU_C930102', 'QU_C930103', 'RISJ880101', 'TUDE900101',
	               'AZAE970101', 'AZAE970102', 'RIER950101', 'WEIL970101', 'WEIL970102', 'MEHP950102', 'KAPO950101',
	               'VOGG950101', 'KOSJ950101', 'KOSJ950102', 'KOSJ950103', 'KOSJ950104', 'KOSJ950105', 'KOSJ950106',
	               'KOSJ950107', 'KOSJ950108', 'KOSJ950109', 'KOSJ950110', 'KOSJ950111', 'KOSJ950112', 'KOSJ950113',
	               'KOSJ950114', 'KOSJ950115', 'OVEJ920102', 'OVEJ920103', 'OVEJ920104', 'OVEJ920105', 'LINK010101',
	               'BLAJ010101', 'PRLA000101', 'PRLA000102', 'DOSZ010101', 'DOSZ010102', 'DOSZ010103', 'DOSZ010104',
	               'GIAG010101', 'DAYM780302', 'HENS920104', 'QUIB020101', 'NAOD960101', 'RUSR970101', 'RUSR970102',
	               'RUSR970103', 'OGAK980101', 'KANM000101', 'NGPC000101', 'MUET010101', 'MUET020101', 'MUET020102',
	               'CROG050101', 'TANS760101', 'TANS760102', 'BRYS930101', 'THOP960101', 'MIRL960101', 'VENM980101',
	               'BASU010101', 'MIYS850102', 'MIYS850103', 'MIYS960101', 'MIYS960102', 'MIYS960103', 'MIYS990106',
	               'MIYS990107', 'LIWA970101', 'KESO980101', 'KESO980102', 'MOOG990101', 'BETM990101', 'TOBD000101',
	               'TOBD000102', 'KOLA930101', 'SKOJ970101', 'SKOJ000101', 'SKOJ000102', 'BONM030101', 'BONM030102',
	               'BONM030103', 'BONM030104', 'BONM030105', 'BONM030106', 'MICC010101', 'SIMK990101', 'SIMK990102',
	               'SIMK990103', 'SIMK990104', 'SIMK990105', 'ZHAC000101', 'ZHAC000102', 'ZHAC000103', 'ZHAC000104',
	               'ZHAC000105', 'ZHAC000106']

	matrix_cols_selected = [e for e in list_columns if e in matrix_cols]

	if matrix_cols_selected and 'Amino_acids' not in list_columns:
		list_columns = list_columns

	if 'Amino_acids' in list_columns:
		list_columns = [e for e in list_columns if e not in matrix_cols_selected]

	thresholds_sup = {
		'CADD_phred': 20,
		'ClinPred_score': 0.5,
		'Condel': 0.49,
		# 'DEOGEN2_score': 0.5,
		# 'fathmm-MKL_coding_score': 0.5,
		'fathmm-XF_coding_score': 0.5,
		'M-CAP_score': 0.025,
		'MetaLR_score': 0.5,
		'MetaSVM_score': 0.82257,
		'PolyPhenVal': 0.8,
		'REVEL_score': 0.5,
		'VEST4_score': 0.5,
		'PrimateAI_score': 0.803,
		'Eigen-raw_coding': 0,
	}
	thresholds_sup_HS = {
		"ClinPred_score": 0.298126307851977,
		"Eigen-raw_coding": -0.353569576359789,
		"M-CAP_score": 0.026337,
		"REVEL_score": 0.235,
		"fathmm-XF_coding_score": 0.22374,
		"PrimateAI_score": 0.358395427465,
		# "MISTIC": 0.277,
		# "MISTIC"    :   0.198003954007379,
	}

	median_dict = {
		"29way_logOdds": 14,
		"CADD_phred": 20,
		"Condel": 0.5,
		"GERP++_RS": 4.8,
		"Grantham": 70,
		"HI_score": 0.25,
		"MetaLR_score": 0.5,
		"MetaSVM_score": 0.82257,
		"phastCons100way_vertebrate": 1,
		"phastCons17way_primate": 0.8,
		"phastCons30way_mammalian": 0.9,
		"phyloP100way_vertebrate": 5.5,
		"phyloP17way_primate": 0.6,
		"phyloP30way_mammalian": 1.1,
		"PolyPhenVal": 0.8,
		"SIFTval": 0.05,
		"VEST4_score": 0.5,
	}

	unwanted = ['', '.', 'NaN', 'nan', 'None', 'NA']

	df['ID'] = df['ID'].str.replace('chr', '')
	df = df[list_columns + flag]

	# df = pd.read_csv(fname, sep='\t', compression='gzip', low_memory=False)
	if 'genename_flag' not in list(df.columns):
		flag += ['genename']
	if 'Source_flag' in list(df.columns):
		flag += ['Source_flag']

	disable = None
	if progress_bar is False:
		disable = True

	# for col in df.columns:

	for col in tqdm(df.columns, disable=disable):
		if 'gnomAD' in col or 'CCRS_score' in col:
			df[col].fillna(0, inplace=True)
		if col in median_dict:
			df[col].fillna(median_dict[col], inplace=True)

		for i, row in enumerate(df[col]):
			if type(row) is str and col != "genename":
				if '[' in row:
					row = ast.literal_eval(row)
					row = [float(e) for e in row if e not in unwanted]
					if 'SIFT' in col:
						row = min(row)
					else:
						row = max(row)
					df.loc[i, col] = row
			if col == 'genename':
				row = str(row)
				if row[0] == '[':
					row = eval(row)[-1]
					df.loc[i, col] = row

	rename_dict = dict()
	# df.dropna(inplace=True)
	for col in df.columns:
		if col in flag:
			if 'Eigen' in col:
				rename_dict[col] = col.replace('_coding', '_coding_flag')
			elif col == 'genename':
				rename_dict[col] = col + '_flag'
			else:
				rename_dict[col] = col.replace('_score', '_flag')

		col_name_pred = col + '_pred'
		if HS is True:
			thresholds_sup = thresholds_sup_HS
		if col in thresholds_sup:
			df[col] = pd.to_numeric(df[col].dropna())
			df[col_name_pred] = np.where(df[col] > thresholds_sup[col], 1, -1)
	df.rename(columns=rename_dict, inplace=True)
	df = df.dropna(subset=list_columns, axis=0).reset_index(drop=True)
	if 'Amino_acids' in list_columns:
		l_matrix = pd.DataFrame(new_columns_matrix(df['Amino_acids'].values))
		if matrix_cols_selected:
			l_matrix = l_matrix[matrix_cols_selected]
		df = pd.concat([df, l_matrix], axis=1)
		list_columns += matrix_cols
		df.drop('Amino_acids', axis=1, inplace=True)

	col_ordered = ['ID', 'True_Label'] + list(sorted(set(list(df.columns)) - {'ID', 'True_Label'}))
	df = df[col_ordered]
	return df

# -*- coding: utf-8 -*-
from pprint import pprint
import logging
import os
import sys
import time
from tqdm import tqdm

start_time = time.time()
logger = logging.getLogger(__name__)

class AAIndex(object):

    def __init__(self):
        self.local_dir = os.path.join('')

        self.data_dir = os.path.join('data/features')
        self.reference_data = None
        self.reference_data_wo_na = None
        self.transtion_scores_wo_na = None

        self.__init_data()

    def __list_files(self, prefix):
        files = [f for f in os.listdir(self.data_dir) if f.startswith(prefix)]
        return files

    def __load_list_files(self, fnames):
        list_files_data = dict()
        for f in tqdm(iterable=fnames, desc='Loading list files', leave=False):
            tmp_fname = f.split('.')[0].split('_')[-1]
            tmp_fname_data = dict()

            with open(file=os.path.join(self.data_dir, f)) as fin:
                for i, line in enumerate(fin):
                    if i > 4:
                        cols = line.strip().split(' ')
                        tmp_matrix_id = cols[0]
                        tmp_matrix_desc = ' '.join(cols[1:])
                        tmp_fname_data[tmp_matrix_id] = {'desc': tmp_matrix_desc,
                                                         'type': tmp_fname}
            list_files_data.update(tmp_fname_data)

        logger.info('Load of list files: DONE')
        return list_files_data

    def __load_matrices(self, fnames):
        matrices_data = dict()
        for i, f in enumerate(tqdm(iterable=fnames, desc='Loading matices', leave=False)):
            if f.endswith('1.txt'):
                tmp_source = f.replace('.txt', '')
                tmp_data = self.__parse_format1(fname=os.path.join(self.data_dir, f))
                matrices_data[tmp_source] = tmp_data
            elif f.endswith('2.txt'):
                tmp_source = f.replace('.txt', '')
                tmp_data = self.__parse_format2(fname=os.path.join(self.data_dir, f))
                matrices_data[tmp_source] = tmp_data
            else:
                tmp_source = f.replace('.txt', '')
                tmp_data = self.__parse_format3(fname=os.path.join(self.data_dir, f))
                matrices_data[tmp_source] = tmp_data

        logger.info('Load of matrices: DONE')
        return matrices_data

    @staticmethod
    def __parse_format1(fname):
        entries = list()
        tmp_entry_data = dict()
        tmp_active_key = None
        i_cnt = 0
        i_rows = {0: ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I'],
                  1: ['L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
                  }

        # Extract entries
        with open(file=fname) as fin:
            for line in fin:
                if line.startswith('//'):
                    i_cnt = 0
                    entries.append(tmp_entry_data)
                    tmp_entry_data = dict()
                else:
                    clean_line = line.rstrip()

                    if clean_line.startswith(' '):

                        for ak in ['A', 'D', 'J', 'T']:
                            if tmp_active_key == ak:
                                tmp_entry_data[ak] += clean_line[1:]

                        if tmp_active_key == 'C':
                            cc = [c.replace(' ', '') for c in clean_line.strip().replace('   ', '=').split('  ')]
                            for c in cc:
                                if len(c) > 1:
                                    k, v = c.split('=')
                                    tmp_entry_data[tmp_active_key].update({k: v})

                        if tmp_active_key == 'I':
                            cols = [m.strip() for m in clean_line[:].strip().split(' ')]
                            scores = list()
                            for c in cols:
                                if c:
                                    clean_c = c if not c.endswith('.') else c + '0'
                                    scores.append(clean_c)
                            for aa, v in zip(i_rows[i_cnt], scores):
                                tmp_entry_data['I'][aa] = v
                            i_cnt += 1

                    else:
                        tmp_active_key = clean_line[0]
                        tmp_active_key_val = clean_line[1:].strip()

                        if tmp_active_key == 'C':
                            tmp_c_data = dict()
                            cc = [c.replace(' ', '') for c in clean_line.strip().replace('   ', '=').split('  ')]
                            for c in cc:
                                if len(c) > 1:
                                    k, v = c.split('=')
                                    tmp_c_data.update({k: v})

                            tmp_active_key_val = tmp_c_data

                        if tmp_active_key == 'I':
                            tmp_active_key_val = dict()

                        tmp_entry_data[tmp_active_key] = tmp_active_key_val

            return entries

    @staticmethod
    def __parse_format2(fname):
        entries = list()
        tmp_entry_data = dict()
        tmp_active_key = None
        m_row = 0
        is_half_matrix = False

        # Extract entries
        with open(file=fname) as fin:
            for line in fin:
                if line.startswith('//'):
                    entries.append(tmp_entry_data)
                    m_row = 0
                    is_half_matrix = False
                    tmp_entry_data = dict()
                else:
                    clean_line = line.rstrip()

                    if clean_line.startswith(' '):
                        for ak in ['A', 'D', 'J', 'T']:
                            if tmp_active_key == ak:
                                tmp_entry_data[ak] += clean_line[1:]

                        if tmp_active_key == 'M':
                            tmp_aa1 = tmp_entry_data['M']['rows'][m_row]
                            cols = [m.strip() for m in clean_line[1:].strip().split(' ')]
                            scores = list()
                            for c in cols:
                                if c:
                                    clean_c = c if not c.endswith('.') else c + '0'
                                    scores.append(clean_c)
                            tmp_aa2 = tmp_entry_data['M']['cols'][:len(scores)]

                            for i, aa2 in enumerate(tmp_aa2):
                                tmp_entry_data['M']['transition']['{}_{}'.format(tmp_aa1, tmp_aa2[i])] = scores[i]

                                # Complement half-matrix
                                if m_row == 0 and len(tmp_aa2) < 20:
                                    is_half_matrix = True

                                if is_half_matrix:
                                    tmp_entry_data['M']['transition']['{}_{}'.format(tmp_aa2[i], tmp_aa1)] = scores[i]
                            m_row += 1
                    else:
                        tmp_active_key = clean_line[0] if clean_line[0] != '*' else 'C'
                        tmp_active_key_val = clean_line[1:].strip()

                        if tmp_active_key == 'C':
                            if 'C' in tmp_entry_data.keys():
                                tmp_active_key_val = tmp_entry_data[tmp_active_key] + '; ' + tmp_active_key_val

                        if tmp_active_key == 'M':
                            rows, cols = tmp_active_key_val.split(',')

                            rows = rows.split('=')[1].strip()
                            cols = cols.split('=')[1].strip()
                            tmp_active_key_val = {'rows': rows,
                                                  'cols': cols,
                                                  'transition': dict()}

                        tmp_entry_data[tmp_active_key] = tmp_active_key_val

        return entries

    @staticmethod
    def __parse_format3(fname):
        entries = list()
        tmp_entry_data = dict()
        tmp_active_key = None
        m_row = 0
        is_half_matrix = False

        # Extract entries
        with open(file=fname) as fin:
            for line in fin:
                if line.startswith('//'):
                    entries.append(tmp_entry_data)
                    m_row = 0
                    is_half_matrix = False
                    tmp_entry_data = dict()
                else:
                    clean_line = line.rstrip()

                    if clean_line.startswith(' '):
                        for ak in ['A', 'D', 'C', 'J', 'T']:
                            if tmp_active_key == ak:
                                tmp_entry_data[ak] += clean_line[1:]

                        if tmp_active_key == 'M':
                            tmp_aa1 = tmp_entry_data['M']['rows'][m_row]
                            cols = [m.strip() for m in clean_line[1:].strip().split(' ')]
                            scores = list()
                            for c in cols:
                                if c:
                                    scores.append(c)
                            tmp_aa2 = tmp_entry_data['M']['cols'][:len(scores)]

                            for i, aa2 in enumerate(tmp_aa2):
                                tmp_entry_data['M']['transition']['{}_{}'.format(tmp_aa1, tmp_aa2[i])] = scores[i]

                                # Complement half-matrix
                                if m_row == 0 and len(tmp_aa2) < 20:
                                    is_half_matrix = True

                                if is_half_matrix:
                                    tmp_entry_data['M']['transition']['{}_{}'.format(tmp_aa2[i], tmp_aa1)] = scores[i]

                            m_row += 1
                    else:
                        tmp_active_key = clean_line[0]
                        tmp_active_key_val = clean_line[1:].strip()

                        if tmp_active_key == 'M':
                            rows, cols = tmp_active_key_val.split(',')

                            rows = rows.split('=')[1].strip()
                            cols = cols.split('=')[1].strip()
                            tmp_active_key_val = {'rows': rows,
                                                  'cols': cols,
                                                  'transition': dict()}

                        tmp_entry_data[tmp_active_key] = tmp_active_key_val

        return entries

    @staticmethod
    def __generate_transition_scores(data, na):
        transition_score = dict()

        # Get transition combinations
        transitions_list = list()
        for d in data:
            if not d.endswith('1'):
                for m in data[d]:
                    transitions_list.extend(list(m['M']['transition'].keys()))
        transitions_list = list(set(transitions_list))

        for t in transitions_list:
            transition_score[t] = dict()

        # Extract scores
        for d in data:
            if d.endswith('1'):
                for m in data[d]:
                    m_id = m['H']

                    if m_id not in na:
                        for aa1 in m['I']:
                            for aa2 in m['I']:
                                aat = '{}_{}'.format(aa1, aa2)
                                tmp_s = float(m['I'][aa2]) - float(m['I'][aa1])
                                transition_score[aat][m_id] = tmp_s
            else:
                for m in data[d]:
                    m_id = m['H']
                    if m_id not in na:
                        for aat in m['M']['transition']:
                            transition_score[aat][m_id] = float(m['M']['transition'][aat])

        logger.info('Generation of transition score: DONE')
        return transition_score

    def __init_data(self):
        matrices_w_na = ['AVBF000101', 'AVBF000102', 'AVBF000103', 'AVBF000104', 'AVBF000105', 'AVBF000106',
                         'AVBF000107', 'AVBF000108', 'AVBF000109', 'YANJ020101', 'GUYH850103', 'ROSM880104',
                         'ROSM880105', 'MEHP950101', 'MEHP950103', 'ROBB790102', 'PARB960101', 'PARB960102',
                         'GODA950101']

        self.reference_data = self.__load_list_files(fnames=self.__list_files(prefix='list'))
        self.reference_data_wo_na = [k for k in self.reference_data.keys() if k not in matrices_w_na]

        matrices_data = self.__load_matrices(fnames=self.__list_files(prefix='aa'))
        self.transtion_scores_wo_na = self.__generate_transition_scores(data=matrices_data, na=matrices_w_na)

    def score_missense(self, aa1, aa2):
        aat = '{}_{}'.format(aa1, aa2)
        return self.transtion_scores_wo_na[aat]

    def get_matrices_desc(self):
        return self.reference_data

    def get_matrix_desc(self, matrix_id):
        return self.reference_data.get(matrix_id)


if __name__ == '__main__':
    # Set logging config
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    a = AAIndex()

    ms = a.score_missense(aa1='R', aa2='A')
    print(list(ms.keys()))

    end_time = time.time()
    sys.stdout.write('Done in {}s\n'.format(round(number=(end_time - start_time), ndigits=3)))

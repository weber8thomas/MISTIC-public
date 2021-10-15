[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/weber8thomas/MISTIC-public/blob/master/LICENSE)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://lbgi.fr/mistic/)
[![GitHub latest commit](https://badgen.net/github/last-commit/Naereen/Strapdown.js)](https://github.com/weber8thomas/MISTIC-public/commit/)


MISTIC-public
==============================


In this repository, you will find all the developped scripts and tools used to build the MISTIC missense prediction model.

## Usage

### Quickstart

#### Conda env

Make sure `conda` is in your shell `$PATH` before :

`make create_environment`

#### Download public & training data from MISTIC

To download GRCh37 files:
- ClinVar VCF (20180930 release)
- training sets data used in MISTIC (TSV file)
- gnomAD VCF (2.1.1 release)

`make dl_data_clinvar_and_training_sets`

> :warning: As gnomAD file is massive, if you already have the raw 2.1.1 file, you can make a symbolic link between your file and the directory data/raw/population. If not, you can use `make dl_data_gnomad` 


#### Prepare data

To use this, you must have access to HGMD data.
HGMD VCF file must be annotated with Variant Effect Predictor (VEP) from Ensembl.

You can specify location of HGMD file with the following :

```
python src/data/handle_config.py --hgmd_file file \
                                 --hgmd_vep_field CSQ \
                                 --hgmd_vep_separator \|
```
> :warning: For VEP separator `|` do not forget the escape character

If you didn't used the `make dl_data`, you can also specify file path, vep field and vep separator for clinvar & gnomad file with this script.

When everything is in place, you can then use :

`make prepare_data`

#### Annotation configuration for VEP & vcfanno

Annotation files from CADD and dbNSFP can be found below : 

- location : `ssh.lbgi.fr:/gstock/biolo_datasets/variation/benchmark/Annot_datasets/dbNSFP/v4.0/dbNSFP_final_hg19.gz`
- https://krishna.gs.washington.edu/download/CADD/v1.4/GRCh37/whole_genome_SNVs.tsv.gz

> **Note** :  dbNSFP was converted to hg19 format and trimmed of some unused columns at the end of the file

To annotate VCF files with both VEP & vcfanno, you can find configuration files used in the development of MISTIC here : `src/annotation`



#### Convert a annotated VCF file to a dataframe (pandas)

To convert an annotated VCF file to a pandas you can use the following : 

```
python src/vcf_to_pandas.py --vcfanno field1 field2 field3
                            --file file.vcf.gz
                            --output file.csv.gz
                            --vep Amino_acids
                            --vep_field CSQ
                            --label 1
```                         

- `vcfanno` field correspond to all numerical vcfanno annotations  
- `vep` field correspond to all vep annotations
- `vep_field` correspond to the name of the vep field  
- `label` correspond to the status of the variant (-1 : benign ; 0 : unknown ; 1 : deleterious)  

#### Train a model

Once your dataframe(s) is/are ready, you use the `MISTIC.py` program.
You can launch here an example with examples files in data/examples with the following: 

`make train`

### Full example

To perform

```
    python MISTIC.py --train_and_test \
                     --input Examples/PANDAS/pandas_mini_training.csv.gz \
                     --output Examples/MODEL_EXAMPLE \
                     --eval Examples/PANDAS/pandas_mini_eval.csv.gz \
                     --list_columns CADD_phred SIFTval VEST4_score gnomAD_exomes_AF \
                     --flag M-CAP_score REVEL_score fathmm-XF_coding_score ClinPred_score \
                     --threads 4

```

### To build synthetic exomes

#### Requirements : 

- bcftools
- bgzip + tabix

> **Note** : 1000 genomes indivudal separated exomes can be download from our server  

1. Download data - location : `ssh.lbgi.fr:/gstock/biolo_datasets/variation/public_genomes/1000G/phase1/individual_vcfs/full/*`

2. Filter MAF - `for file in *.vcf.gz; bcftools view -i 'INFO/AF < 0.01' "$file".vcf.gz | bgzip > "${file%%.*}"_bcftools.vcf.gz`

3. Annotate VEP  - `for file in *_bcftools.vcf.gz; vep -i "$file".vcf.gz -o "${file%%.*}"_vep.vcf.gz`

4. Filter VEP with missenses only - `for file in *_vep.vcf.gz; python filter_1000G.py -i $file.vcf.gz -o "${file%%.*}"_missense.vcf.gz`

5. Annotate vcfanno - `for file in *_missense.vcf.gz; vcfanno -p core_nb conf.toml $file.vcf.gz | bgzip > "${file%%.*}"vcfanno.vcf.gz`

6. Convert to pandas - `for file in *_vcfanno.vcf.gz; vcf_to_pandas.py $ARGS` (see above)

7. Merge pandas & score with MISTIC - `python merge_and_score_1000G.py $input_dir_1000G $output_dir` 

8. Compute stats & produce plots - `stats_1000G.py $1000G_processed_dir`
 

## Project Organization


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── examples       <- Pandas examples for learning.
    │   ├── features       <- Intermediate data that has been transformed.
    │   ├── processed      <- Final VCF data after filtering.
    │   └── raw            <- Raw files.
    │
    │── docs               <- Docs directory
    │
    │── models             <- Model directory
    │
    │── outputs            <- Output directory
    │
    ├── .MISTIC-public.yml  <- The requirements file for reproducing the analysis environment
    |
    ├── MISTIC.py          <- The main program of this project
    │
    ├── src                <- Source code for use in this project.
        │── annotation     <- annotation files
        │   └── conf.toml
        │   └── vep.ini
        │
        ├── data           <- Scripts to download or generate data
        │   └── filter_1000G.py
        │   └── handle_config.py
        │   └── make_dataset.py
        │   └── merge_and_score_1000G.py
        │   └── vcf_to_pandas.py
        |
        │── evaluation     <- Scripts to build evaluation sets
        │   └── combination_pandas.py        |
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── AAIndex.py
        │   └── select_columns_pandas.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── ML.py
        │   └── training.py
        │   └── testing.py
        │
        ├── utils          <- Scripts to turn raw data into features for modeling
        │   └── utils.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── histo_weights.py
            └── maf_plot.py
            └── stats_1000G.py
            └── visualization.py


--------



### Note (Windows Users)

If you are running under windows, `cyvcf2` can't be installed. Scripts which process VCF files (`vcf_to_pandas.py`, `filter_1000G.py`, `make_dataset.py`, ...) could not be used.
Fortunately, you can still use `MISTIC` be installing the required packages with the following command : 

```
conda create -c conda-forge -n MISTIC-public matplotlib==2.2.3 numpy pandas==0.23.4 scikit-learn==0.20.2 seaborn==0.9.0 tqdm parmap
```                            

## Contact

Feel free to post issues if you have troubles


--------------------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## How to cite ? 

Chennen K, Weber T, Lornage X, Kress A, Böhm J, Thompson J, Laporte J, Poch O. MISTIC: A prediction tool to reveal disease-relevant deleterious missense variants. 2020 Jul 31;15(7):e0236962. doi: https://doi.org/10.1371/journal.pone.0236962. eCollection 2020.


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
- gnomAD VCF (2.1.1 release)
- training sets data used in MISTIC (TSV file)

`make dl_data`

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
    ├── outputs            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── MISTIC-public.yml  <- The requirements file for reproducing the analysis environment
    ├── MISTIC.py          <- The requirements file for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │   └── vcf_to_pandas.py
        |
        │── evaluation     <- Scripts to download or generate data
        │   └── combination_pandas.py
        │
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
            └── visualization.py


--------



### Note (Windows Users)

If you are running under windows, `cyvcf2` can't be installed. Scripts which process VCF files (`vcf_to_pandas.py`, `compare_vcf.py`, ...) could not be used.
Fortunately, you can still use `MISTIC` be installing the required packages with the following command : 

```
conda create -c conda-forge -n MISTIC-public matplotlib==2.2.3 numpy pandas==0.23.4 scikit-learn==0.20.2 seaborn==0.9.0 tqdm
```                            

## Contact

Feel free to post issues if you have troubles


--------------------
#<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

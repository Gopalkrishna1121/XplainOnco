# Data layout for TCGA-BRCA

`src/xplainonco/data/tcga.py` reads one folder of tab-separated tables. **It has only been tested on small files
written by `tests/test_data.py`, not on real downloads.** Real files differ in small ways (sample-id suffixes, gene
aliases, header rows), so expect to adjust it the first time.

| File | Rows x columns | Content |
|---|---|---|
| `expression.tsv` | genes x samples | RNA-seq. Raw counts (run with `--counts`) or log2(count+1) |
| `mutation.tsv` | genes x samples | 1 for a non-silent somatic mutation, otherwise 0. Genes with no row are treated as 0 |
| `cna.tsv` | genes x samples | GISTIC2 thresholded copy-number level, -2 to 2 |
| `labels.tsv` | samples x columns | A `subtype` column (`LumA`, `LumB`, `Her2`, `Basal`, `Normal`) and optional numeric clinical columns such as `age` |
| `string_links.tsv` | one row per link | `gene_a`, `gene_b`, `combined_score` (0 to 1000), already mapped to gene symbols |
| `pam50.txt` | optional | One gene symbol per line, the genes that define the labels |

Gene symbols must agree across the files. Samples are matched on the 12-character TCGA patient barcode, and a
patient missing expression, mutation, copy number or a label is dropped.

## Where the files usually come from

- Expression, mutation, copy number, subtype and age: GDC / cBioPortal / UCSC Xena TCGA-BRCA (PanCancer Atlas).
  The PAM50 call in the Xena and TCGA clinical tables is what the paper used as the label.
- STRING: `9606.protein.links.v12.0.txt` plus the protein info file to map Ensembl protein ids to gene symbols.

## Run

```bash
python scripts/run_cv.py --data path/to/folder             # paper settings: 5 folds x 3 repeats
python scripts/run_cv.py --data path/to/folder --withhold-label-genes   # PAM50-withheld experiment
```

All gene selection (ANOVA F-test with Benjamini-Hochberg, top-decile variance, STRING score of at least 700 and
connectivity), standardisation and the Integrated Gradients baseline are fitted inside each training fold.
Full settings need a GPU and many hours; use `--quick` first to check that your files load and the pipeline runs.

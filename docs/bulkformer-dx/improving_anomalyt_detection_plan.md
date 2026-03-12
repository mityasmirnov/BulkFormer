# Cursor Prompt: BulkFormer Outlier Detection Benchmarking Plan Inspired by OUTRIDER and TabPFN

## Core references and what they imply for implementation choices

You are implementing and benchmarking **multiple outlier/anomaly scoring + statistical testing** variants for bulk RNA-seq, using **BulkFormer** as the predictive engine, and using OUTRIDER- and TabPFN-inspired probability tests.

### How OUTRIDER performs the statistical test

OUTRIDER’s *conceptual* outlier test is:

- Learn **expected read counts** per gene per sample (using an autoencoder that captures gene–gene covariation / hidden confounders).
- Assume **observed counts follow a Negative Binomial** distribution around that expected mean with a **gene-specific dispersion**.
- Compute **p-values** for each (sample, gene) count and apply **multiple testing correction** (OUTRIDER commonly uses Benjamini–Yekutieli because expression tests are dependent/correlated). citeturn23search1turn23search6turn23search13

The key part you must reproduce/adapt is the **two-sided NB p-value computation** (discrete-safe) used in OUTRIDER’s code:

- Let \(k\) be observed count, and \(X \sim NB(\mu, \theta)\) (parameterization via mean \(\mu\), size \(\theta\)).
- Compute:
  - \(p_{\le} = P(X \le k)\)
  - \(p_{=} = P(X = k)\)
  - \(p_{\ge} = P(X \ge k) = 1 - p_{\le} + p_{=}\)
- OUTRIDER’s **two-sided** p-value is:
  \[
  p_{2s} = 2 \cdot \min(0.5, p_{\le}, p_{\ge})
  \]
  which clamps and handles discreteness. citeturn3search0

OUTRIDER’s `computePvalues` documentation emphasizes that per-sample multiple testing adjustment and sidedness options exist, with BY used in the ecosystem for correlated gene tests. citeturn1search31turn23search13

### What BulkFormer gives you

BulkFormer is trained with a **masked reconstruction** objective: roughly **15%** of gene expression values are masked with a special token (reported as **−10**) and the model is trained to reconstruct masked values with an **MSE loss**. citeturn13search7turn12search1

This happens in **continuous expression space**, not NB count likelihood space. Therefore:

- The *native* BulkFormer output is a **point prediction** \(\hat y\) (e.g., for `log1p(TPM)` in your DX pipeline).
- To do **statistical testing**, you must add/estimate **uncertainty**:
  - from a learned head (heteroscedastic regression),
  - from cohort-calibrated residual dispersion,
  - from MC masking variability,
  - or by mapping predictions back to **count space** and using an NB model (OUTRIDER-like).

### What TabPFN’s unsupervised outlier detection suggests

TabPFN’s unsupervised extension frames outlier detection as **density estimation**:

- It decomposes the joint feature density via the chain rule:
  \[
  P(X) = \prod_{i=1}^{d} P(X_i \mid X_{<i})
  \]
- It computes a **sample log-likelihood score** by summing log-probabilities across features and **averaging over random permutations** to stabilize dependence on feature ordering. Samples with low probability (low log-likelihood) are outliers. citeturn20search0turn21search2

BulkFormer is not autoregressive, but it *is* a masked conditional predictor. That means you can implement the **TabPFN idea** via:
- a **pseudo-likelihood** or **MC-masking likelihood** surrogate: estimate \(\log p(y_g \mid \text{context})\) for many masked contexts and aggregate.

## What you already implemented in bulkformer_dx and what’s missing

You already have a strong baseline toolkit in `mityasmirnov/BulkFormer`:

- Preprocess: counts → TPM → `log1p(TPM)`, align genes, fill missing with `-10`.
- Outlier scoring: Monte Carlo masking over valid genes + reconstruction residuals (mean abs residual as `anomaly_score`).
- Cohort calibration:
  - empirical cohort-tail p-values on `anomaly_score`,
  - BY correction within each sample,
  - plus an explicit **normalized outlier** table using \(z = (Y-\mu)/\sigma\), **normal** two-sided p-values, BY correction, and significance calls.
- Optional `nb_approx` is explicitly labeled as an approximation (it currently operates in TPM-derived pseudo-count space and estimates dispersion from TPM variance).  

What’s missing (and what you must improve/benchmark):

- A **true OUTRIDER-style NB test in count space**, driven by BulkFormer’s expected expression, with **proper count mean mapping** and **dispersion estimation/shrinkage** grounded in count statistics.
- A **TabPFN-style likelihood scoring** variant (gene-wise conditional densities aggregated into sample outlier scores), not just residual magnitude.
- A systematic **benchmark harness** that compares:
  - residual ranking vs likelihood ranking,
  - Gaussian vs Student-t vs NB,
  - global-cohort vs local-cohort (embedding-neighborhood) calibration,
  - N=1 use vs cohort use,
  - and checks whether p-values behave like p-values (calibration diagnostics).

## Methods to implement and benchmark

Implement all methods behind a unified interface so you can run them in a grid and benchmark them fairly.

### Residual-based baselines to keep

Keep as baselines (they’re fast and often strong in practice):

- **Residual magnitude**: mean absolute residual over MC masks (`anomaly_score` in current scoring).
- **Gaussian z-score** using cohort-estimated sigma: \(z=(Y-\mu)/\hat\sigma_g\), \(p = 2\Phi(-|z|)\), then BY within sample.

These are the anchor baselines for comparisons.

### OUTRIDER-style NB test with BulkFormer as the mean model

Goal: replicate OUTRIDER’s *statistical testing step* but swap the mean model (autoencoder) with BulkFormer.

**Data needed**: raw counts \(K_{jg}\), gene length \(L_g\) (bp or kb), and BulkFormer predicted expression in TPM space.

**Key mapping**

You currently have TPM computed from counts via:

- \( r_{jg} = K_{jg} / L^{kb}_g \)
- \( TPM_{jg} = r_{jg} / \sum_h r_{jh} \cdot 10^6 \)

Let \(S_j = \sum_h r_{jh}\) computed from the observed counts.

If BulkFormer gives an expected \( \widehat{TPM}_{jg} \), you can derive an expected count mean:

\[
\widehat{\mu}^{count}_{jg} = \widehat{TPM}_{jg} \cdot \frac{S_j}{10^6} \cdot L^{kb}_g
\]

Then model:
\[
K_{jg} \sim NB(\mu = \widehat{\mu}^{count}_{jg},\; \theta_g\ \text{or}\ \alpha_g)
\]

and compute p-values using the OUTRIDER two-sided formula (discrete-safe). citeturn3search0turn23search13

**Dispersion estimation**

Use DESeq2’s NB mean–variance relationship as a guiding reference:

\[
Var(K_{ij}) = \mu_{ij} + \alpha_i \mu_{ij}^2
\]
citeturn22search6

Also, DESeq2’s parametric mean–dispersion trend is:
\[
\alpha(\bar\mu) = asymptDisp + \frac{extraPois}{\bar\mu}
\]
citeturn22search37turn22search0

Implement at least two dispersion strategies and benchmark them:

- **Per-gene MLE** of \(\alpha_g\) given \(\mu_{jg}\) (optimize NB log-likelihood over \(\alpha_g>0\)).
- **Shrinkage to trend**:
  1. estimate per-gene \(\alpha_g^{MLE}\),
  2. fit a trend \(\alpha(\bar\mu_g)\) (parametric or local),
  3. shrink extreme/noisy genes toward the trend (simple convex combo in log-space is fine for a first version).

Also implement **robust fitting** options (e.g., trimming top residuals) because outliers inflate dispersion and destroy sensitivity.

### TabPFN-like likelihood scoring adapted to BulkFormer

TabPFN computes a sample-level outlier score as (averaged-permutation) joint likelihood via chain rule. citeturn20search0turn21search2

BulkFormer can approximate this via masked conditional prediction, yielding either:

- **Pseudo-likelihood**: \( \sum_g \log p(y_g \mid y_{-g}) \)  
- **MC masked-likelihood**: over many random mask patterns \(m\), treat masked genes as “targets” and estimate:
  \[
  score(sample) = -\frac{1}{|\mathcal{M}|}\sum_{(m,g)\in \mathcal{M}} \log p(y_g \mid y_{\neg g}^{(m)})
  \]
  where each pass masks a subset and you accumulate log-probabilities for the masked positions.

Critical: you must choose \(p(\cdot)\) (the conditional density) to turn point predictions into probabilities:

Implement at least these density options:
- **Gaussian in log1p(TPM) space**: \(y \sim \mathcal{N}(\mu=\hat y,\sigma)\)
  - \(\sigma\) from your existing sigma head,
  - or \(\sigma\) from cohort residuals,
  - or \(\sigma\) from MC variance in predictions across masks (epistemic-ish).
- **Student-t in log1p(TPM) space** (heavier tails; often better calibrated for residuals than Gaussian).
- **NB in count space** as above (if counts are available).

Then:

- For each gene, compute gene-level **NLL contribution** and output:
  - per-gene NLL,
  - per-sample aggregated NLL,
  - plus p-values if you transform NLL into tail probabilities via cohort calibration.

### Cohort selection and confounder control

OUTRIDER’s advantage comes from modeling covariation/confounders with an autoencoder. citeturn23search1  
BulkFormer already captures strong transcriptome structure, but your **calibration cohort** can still dominate results.

Implement cohort selection variants:

- **Global cohort calibration**: use all samples (current behavior).
- **Local cohort calibration**: for each sample, find nearest neighbors in BulkFormer **sample embedding space**, and calibrate sigma/dispersion only on those neighbors.
  - This is your practical substitute for confounder correction when cohorts are heterogeneous.

Benchmark local vs global.

## Implementation plan as a step-by-step TODO list for Cursor

You are working inside `mityasmirnov/BulkFormer` and must integrate with the existing `bulkformer_dx` package and CLI.

### Step one: Establish a reproducible benchmark harness and reporting skeleton

**Goal**: before modifying any stats, create an evaluation scaffold so every future change produces comparable metrics + plots.

TODO
- Add `bulkformer_dx/benchmark/` module with:
  - `datasets.py`: loaders for (a) user-provided cohort counts+annotation+metadata, (b) synthetic simulator.
  - `inject.py`: inject controlled outliers (up/down) into either log1p(TPM) or counts; record ground truth mask.
  - `runner.py`: run a method config grid and write standardized outputs.
  - `metrics.py`: AUROC, AUPRC, recall@FDR, precision@k, calibration metrics (see below).
  - `plots.py`: matplotlib-only plots.
- Add CLI entry:
  - `python -m bulkformer_dx.cli benchmark run ...`
- Add a single JSON-serializable “method config” schema:
  - `method_name`
  - `space` in {`log1p_tpm`, `counts`}
  - `uncertainty_source` in {`cohort_sigma`, `sigma_head`, `mc_variance`, `nb_dispersion`}
  - `test` in {`gaussian_z`, `student_t_z`, `nb_outrider_2s`, `empirical_tail`, `mc_pseudolikelihood`}
  - `multiple_testing` in {`BY`, `BH`, `none`}
  - `cohort_mode` in {`global`, `knn_local(k=...)`}
- Add a `reports/` folder with a template `reports/README.md` describing the required artifacts and naming scheme.

Reports required for this step
- `reports/step_one_benchmark_scaffold.md` containing:
  - A diagram of the benchmark pipeline (box diagram is fine).
  - A table listing all planned method configs to be benchmarked.
  - A “smoke test” run on synthetic data showing the harness produces:
    - metrics JSON,
    - per-method ranked outputs,
    - at least one figure saved to disk.

### Step two: Extend preprocessing to export aligned count-space artifacts

**Goal**: enable true NB on counts, not TPM-rounded pseudo-counts.

TODO
- Modify `bulkformer_dx/preprocess.py` to additionally export:
  - `aligned_counts.tsv` (samples × BulkFormer gene panel; missing genes = NaN or 0 but must be flagged invalid via mask)
  - `aligned_tpm.tsv` (optional; same alignment)
  - `gene_lengths_aligned.tsv` (BulkFormer gene panel with length_kb, and a “has_length” flag)
  - `sample_scaling.tsv` with \(S_j = \sum_h K_{jh}/L^{kb}_h\) computed on observed genes (needed for TPM↔counts mapping)
- Ensure you do **not** fill counts with the BulkFormer mask token `-10`. Counts missing genes should not masquerade as real 0 counts; keep them explicit via mask/NaN.
- Update docs to state: NB testing only applies to `is_valid==1` genes in `valid_gene_mask.tsv`.

Reports required for this step
- `reports/step_two_preprocess_counts.md` with:
  - QC plots:
    - distribution of library sizes and \(S_j\),
    - histogram of gene lengths used,
    - fraction of BulkFormer-valid genes per sample.
  - A “sanity check” table for 5 random genes showing:
    - counts, TPM, log1p(TPM), and the derived mapping terms.

### Step three: Implement OUTRIDER-style NB p-values using BulkFormer mean predictions

**Goal**: implement the *actual* OUTRIDER statistical testing step (two-sided discrete-safe NB p-values), but using BulkFormer for \(\mu_{jg}\).

TODO
- Add `bulkformer_dx/anomaly/nb_test.py` implementing:
  - `expected_counts_from_predicted_tpm(pred_tpm, counts, gene_lengths_kb, S_j)` implementing:
    \[
    \widehat{\mu}^{count}_{jg} = \widehat{TPM}_{jg} \cdot \frac{S_j}{10^6} \cdot L^{kb}_g
    \]
  - `fit_nb_dispersion(...)`:
    - implement `dispersion_method="mle"` and `dispersion_method="moments_regression"` at minimum.
    - optional `dispersion_shrinkage="deseq2_parametric"` that fits
      \[
      \alpha(\bar\mu) = asymptDisp + extraPois/\bar\mu
      \]
      for stabilization, inspired by DESeq2. citeturn22search37turn22search0
  - `outrider_two_sided_nb_pvalue(k, mu, size)` using:
    \[
    p_{2s} = 2 \cdot \min(0.5, P(X \le k), P(X \ge k))
    \]
    with \(P(X \ge k)=1-P(X\le k)+P(X=k)\). citeturn3search0
- Extend `bulkformer_dx/anomaly/calibration.py`:
  - Add `count_space_method="nb_outrider"` alongside existing `none` and `nb_approx`.
  - `nb_outrider` must:
    - load `aligned_counts.tsv`, `gene_lengths_aligned.tsv`, `sample_scaling.tsv`,
    - use BulkFormer predictions (already in ranked tables as `mean_predicted_expression` in log1p(TPM)) to get `pred_tpm`,
    - map to expected counts mean,
    - load or fit dispersions,
    - compute per-gene p-values (one- and two-sided),
    - apply BY per sample (default) like OUTRIDER/DROP conventions. citeturn23search13turn1search31
- Add caching:
  - dispersion fitting is expensive; cache gene-wise dispersion estimates + trend parameters in `output_dir/nb_params.json` (or parquet).
- Add unit tests:
  - verify p-value formula matches OUTRIDER’s discrete-safe behavior on small hand-checked examples.
  - verify NB p-values are uniform-ish under null synthetic generation.

Reports required for this step
- `reports/step_three_nb_outrider_test.md` with:
  - Dispersion diagnostics:
    - plot of gene-wise \(\alpha_g\) vs mean count,
    - trend fit overlay (if shrinkage enabled).
  - P-value calibration diagnostics on null synthetic data:
    - histogram of p-values (should be ~uniform),
    - QQ plot of p-values (expected line),
    - KS statistic against Uniform(0,1) for multiple samples.
  - Power diagnostics on injected outliers:
    - AUPRC/AUROC for detecting injected gene outliers,
    - recall at BY-FDR 0.05 and 0.1,
    - compare “Gaussian z” vs “NB OUTRIDER” directly.

### Step four: Implement TabPFN-style likelihood scoring using BulkFormer

**Goal**: create a “foundation-model density” outlier score by aggregating conditional log-probabilities across genes, analogous to TabPFN’s unsupervised outliers. citeturn20search0turn21search2

TODO
- Add `bulkformer_dx/anomaly/likelihood.py` implementing:
  - `mc_masked_loglikelihood_score(...)`:
    - re-use existing MC mask plan generation (from `scoring.py`),
    - for each pass, compute BulkFormer predictions for masked genes,
    - compute per-masked-gene log-prob using selected distribution:
      - Gaussian(log1pTPM) with sigma source,
      - StudentT(log1pTPM) with dof parameter,
      - NB(counts) if count-space artifacts provided.
    - aggregate:
      - per-gene: mean NLL over all times the gene was masked,
      - per-sample: mean NLL over all masked gene predictions.
  - Add `uncertainty_source` options:
    - `cohort_sigma` (existing MAD-based sigma per gene),
    - `sigma_head` (load and apply your sigma head to gene embeddings; use its predicted sigma),
    - `mc_variance` (use variance of predicted values across masks as sigma proxy).
- Extend CLI `anomaly score`:
  - allow `--score-type residual | nll`
  - emit ranked tables with:
    - `nll_score`, `log_prob`, `sigma_used`, etc.
- Extend calibration:
  - for NLL scores, provide empirical cohort-tail p-values per gene (like you do for residual-based anomaly_score).
  - Optional: convert NLL into z-like standardized residuals when Gaussian is used.

Reports required for this step
- `reports/step_four_tabpfn_like_likelihood.md` with:
  - Explanation of how TabPFN computes outlier likelihood via chain rule and permutation averaging, and what the BulkFormer analog is (pseudo-likelihood / MC masked-likelihood). citeturn20search0turn21search2
  - Comparisons (same injected dataset) across:
    - residual score vs NLL score,
    - sigma source variants (cohort_sigma vs sigma_head vs mc_variance),
    - stability vs MC passes (plot performance vs passes).

### Step five: Local cohort calibration using BulkFormer embeddings

**Goal**: reduce confounding by calibrating against “similar” samples (tissue/state matched) using BulkFormer sample embeddings.

TODO
- Reuse `bulkformer_dx/bulkformer_model.py` embedding extraction utilities.
- Implement `bulkformer_dx/anomaly/cohort.py`:
  - compute sample embeddings for the cohort once,
  - for each target sample, select k nearest neighbors,
  - compute gene-wise sigma (or dispersion) from only those neighbors,
  - run the chosen test (Gaussian/NB) and BY correction.
- Extend calibration CLI:
  - `--cohort-mode global|knn_local`
  - `--knn-k 50` etc.
- Benchmark: does local calibration reduce false positives driven by tissue differences?

Reports required for this step
- `reports/step_five_local_cohort.md` with:
  - UMAP/PCA of sample embeddings colored by known metadata if available, otherwise by simple QC stats.
  - For each calibration mode:
    - number of significant genes per sample distribution,
    - overlap/Jaccard of top outliers,
    - injected benchmark performance.
  - A “failure mode” section: when local cohort is too small or mixed, show how p-value calibration breaks.

### Step six: Unified benchmark matrix and final comparative report

**Goal**: one command produces a full benchmark comparing every method, including your current implementation.

TODO
- Implement `bulkformer_dx.cli benchmark grid-run` that:
  - reads a YAML/JSON config listing methods,
  - runs all methods with fixed seeds,
  - writes a single `benchmark_results.parquet` and `benchmark_summary.json`,
  - writes plots into `benchmark_figures/`.
- Include these method families in the default grid:
  - Current: residual + empirical tail + BY
  - Normal z-score (cohort sigma) + BY
  - Student-t z-score (cohort sigma) + BY
  - NB OUTRIDER-style (counts) + BY
  - NLL pseudo-likelihood (Gaussian; sigma_head) + empirical calibration + BY
  - Local-cohort variants of at least the top 2 performing methods
- Add “distribution misfit detectors”:
  - p-value uniformity tests under null,
  - residual QQ plots for Gaussian/t,
  - NB Pearson residual histogram.

Reports required for this step
- `reports/final_benchmark_report.md` with:
  - A single leaderboard table for gene-level detection (AUPRC primary, AUROC secondary).
  - Calibration table:
    - KS statistic on null p-values,
    - fraction of samples with inflated significant genes.
  - Compute/runtime table (seconds, GPU memory if available).
  - Figures:
    - PR curves for the key methods,
    - QQ plots for p-values (Gaussian vs NB),
    - dispersion trend plot for NB method,
    - ablation: performance vs MC passes.

## Acceptance criteria and Ralph workflow integration

You must implement this using the repo’s **Ralph workflow** (fresh-context iterations + external verification), so the loop stops only when tests and benchmark checks pass.

Acceptance criteria
- All new CLI commands have:
  - `--help` works,
  - deterministic outputs with fixed seeds,
  - clear artifact schemas.
- Unit tests:
  - cover NB p-value function (discrete-safe),
  - cover counts↔TPM expected-count mapping,
  - cover calibration uniformity on null synthetic datasets.
- Bench harness:
  - runs end-to-end on synthetic data in CI-like time (small synthetic cohorts),
  - writes a complete report bundle.
- Documentation:
  - updated `docs/bulkformer-dx/anomaly.md` to describe new methods and when to use them.

Ralph instructions
- Update `scripts/ralph/prd.json` to include user stories for each step above.
- For each story:
  - implement code + tests,
  - add/extend docs,
  - add the step report markdown in `reports/`,
  - ensure `pytest` passes.
- Run:
  - `./scripts/ralph/ralph.sh <N>` until external verification passes (repo standard).

## Notes on “best distribution” decision rule

You will not pick the “best” distribution by intuition; you will pick it by **calibration + power**:

- If raw counts are available, NB is the most principled for RNA-seq count noise, with variance \(\mu + \alpha \mu^2\). citeturn22search6turn23search13  
  But it only wins if dispersion estimation is stable and p-values are calibrated.
- If only normalized continuous values exist (TPM/logTPM), Gaussian or Student-t in log space can work well, but you must validate p-value behavior (QQ + KS) and tail behavior (false positives).
- TabPFN-style likelihood scoring should be evaluated primarily for:
  - ranking quality (AUPRC),
  - stability vs masking/permutations,
  - and usefulness in N=1 contexts (does it still highlight the injected genes?) citeturn20search0turn21search2
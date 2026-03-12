> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.
>
> 1 **A** **large-scale** **foundation** **model** **for** **bulk**
> **transcriptomes**
>
> 2 Boming Kang1#, Rui Fan1#, Meizheng Yi2, Chunmei Cui3 and Qinghua
> Cui1,3\*
>
> 3 1Department ofBiomedical Informatics, State Key LaboratoryofVascular
> HomeostasisandRemodeling,
>
> 4 School of Basic Medical Sciences, Peking University, 38 Xueyuan Rd,
> Beijing, 100191, China.
>
> 5 2School of Pharmaceutical Sciences, Jilin University, Changchun
> 130021, China.
>
> 6 3School of Sports Medicine, Wuhan Institute of Physical Education,
> No. 461 Luoyu Rd. Hongshan
>
> 7 District, Wuhan 430079, Hubei Province, China
>
> 8 \#These authors contributed equally to this work
>
> 9 \*To whom the correspondence should be addressed:

10 Qinghua Cui, email: <cuiqinghua@bjmu.edu.cn>

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

11 **Abstract**

12 Large language models (LLMs) have emerged as powerful foundation
models leading to breakthroughs

13 in transcriptome analysis. However, current RNA-seq foundation models
are exclusively pretrained on

14 sparse single-cell RNA-seq (scRNA-seq) data, which typically detects
only ~3000 genes per cell. This

15 thus creates a critical gap in models specifically designed for bulk
transcriptomes, a fundamentally

16 different modality capable of profiling ~16,000 genes per sample.
Here we propose BulkFormer, a large-

17 scale foundation model for bulk transcriptome analysis. With 150
million parameters covering about

18 20,000 protein-coding genes, BulkFormer is pretrained on over 500,000
human bulk transcriptomic

19 profiles. BulkFormer incorporates a hybrid encoder architecture,
combining a graph neural network to

20 capture explicit gene-gene interactions and a performer module to
model global expression

21 dependencies. As a result, despite incurring much lower training
costs than scRNA-seq foundation

22 models, BulkFormer consistently outperforms them in all six
downstream tasks: transcriptome

23 imputation, disease annotation, prognosis modeling, drug response
prediction, compound perturbation

24 simulation, andgene essentiality scoring. Notably, BulkFormer not
only enhances the discoveryof novel

25 clinical biomarkers but also uncovers latent disease mechanisms by
imputing biologically meaningful

26 gene expression. Collectively, these results demonstrate BulkFormer’s
power as a versatile and robust

27 framework for bulk transcriptome modeling and biomedical discovery,
bridging a critical gap in the

28 current foundation model landscape.

29

30

31

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

32 **Introduction**

33 Large-scale pretrained language models represent a revolutionary
breakthrough in the field of natural

34 language processing (NLP) in recent years**1**. Similar to natural
language, DNA, RNA, and protein

35 sequences in life sciences can also be regarded as biological
languages, leading to the development of a

36 series of large-scale pretrained biological language models**2**,
such as DNA-BERT**3**, RNA-FM**4**, and

37 ESM2**5**. Unlike biological sequences, gene expression profiles
derived from transcriptomics encode the

38 biological information of life systems, serving as a functional
language that reflects the physiological

39 state of the living organisms**6**. For example, patient prognosis
can be predicted using the expression

40 patterns of disease-associated biomarkers**7**. Transcriptomic
sequencing technologies can be broadly

41 classified into bulk RNA sequencing and single-cell RNA sequencing
(scRNA-seq). Bulk RNA-seq

42 measures the average gene expression across a population of cells,
thus providing a global but low-

43 resolution view of transcriptional activity. In contrast, scRNA-seq
captures gene expression at single-

44 cell resolution, enabling the identification of cellular
heterogeneity and rare cell types. Therefore, the

45 substantially larger scale of gene expression data generated by
scRNA-seq compared to bulk RNA-seq

46 has driven the development of a series of foundation models
specifically pretrained on single-cell

47 transcriptomic data, including Geneformer**8**, scGPT**9**,
scFoundation**10**, GeneCompass**11**, and scLong**12**.

48 Single-cell large language models (scLLMs) have demonstrated the
ability to extract high-quality

49 cellular and gene-level transcriptomic representations, enabling
state-of-the-art performance in diverse

50 downstream single-cell tasks such as cell annotation, drug response
prediction, perturbation effect

51 prediction, and gene module inference**13,14.** Although scRNA-seq
provides single-cell resolution, its

52 inherent sparsity, defined as the limited detection of gene
expression per cell**15**, presents difficulties for

53 downstream tasks requiring comprehensive transcriptomic coverage,
such as disease subtype

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

54 classification and prognostic modeling**16**. In contrast, bulk
RNA-seq offers more comprehensive and

55 stable gene expression measurements across samples, making it
well-suited for system- or tissue-level

56 analyses. However, large-scale models pretrained on bulk
transcriptomic data still remain unavailable,

57 highlighting a critical gap in the fields of transcriptomic modeling.

58 Given the characteristics of scRNA-seq data, existing scLLM models
have adopted various training

59 strategies. Geneformer ranks normalized gene expression values within
each cell andis trained to predict

60 the rank value of masked genes. scGPT discretize gene expression into
bins and predict the bin

61 membership of masked genes. While these approaches help mitigate data
noise, they reduce the

62 resolution of expression modeling and may impair performance in
downstream tasks. To address this

63 limitation, scFoundation and scLong directly predict the continuous
expression values of masked genes,

64 thereby improving modeling resolution. Distinctively, GeneCompass
employs a multitask pretraining

65 strategy with a dual-decoder architecture: one decoder predicts the
gene identity at masked positions,

66 and the other predicts the corresponding expression values. However,
the encoder inputs of these models

67 typically include only the top expressed few thousand genes in each
sample, while the remaining large-

68 number of genes, which are failed to be detected due to technical
limitations, are assigned zero

69 expression values. Although this strategy is appropriate for handling
the sparsity of scRNA-seq data, it

70 prevents the model from learning the complete set of gene-gene
relationships across the whole

71 transcriptome. As a result, these scLLMs are not well suited for bulk
RNA-seq data and its associated

72 downstream tasks.

73 In this study, we focused on bulk RNA-seq modeling and proposed
BulkFomer, a large-scale foundation

74 model with approximately 150 million parameters, covering around
20,000 protein-coding genes. To

75 enable large-scale pretraining, we curated and standardized
approximately 520,000 bulk RNA-seq gene

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

76 expression profiles from public databases. To more effectively model
bulk RNA-seq data, we developed

77 a hybrid encoder architecture that integrates both graph neural
networks (GNN) for capturing explicit

78 gene-gene relationships from a biological knowledge graph while
employing attention mechanisms to

79 learn implicit transcriptional dependencies across the entire
transcriptome.

80 To verify BulkFormer’s power, we performed extensive benchmarking
across six critical downstream

81 tasks, including transcriptome imputation, disease annotation,
prognosis modeling, drug response

82 prediction, compound perturbation prediction, and gene essentiality
prediction. As a result, BulkFormer

83 outperforms existing scLLMs in all tasks. Notably, when applied to
clinical samples, BulkFormer

84 successfully reconstructed missing gene expression values, enabling
the discovery of a series of

85 previously unrecognized prognostic biomarkers. These results
collectively establish BulkFormer as a

86 powerful and versatile tool for bulk RNA-seq modeling and analysis.
This work not only advances the

87 development of foundation models for bulk transcriptomics but also
opens new avenues for their

88 biomedical applications.

89 **Results**

90 **Overview** **of** **BulkFormer**

91 To construct a large-scale dataset for BulkFormer pretraining, we
first curated PreBULK, a

92 comprehensive bulk transcriptomic dataset assembled from public
repositories including Gene

93 Expression Omnibus (GEO)**17** andARCHS4**18**, comprising 522,769
gene expression profiles for 20,010

94 protein-coding genes. BulkFormer was pretrained using a masked
language modeling (MLM) strategy

95 inspired by BERT**19**, in which 15% of gene expression values in the
input transcriptome are randomly

96 masked. The model is then trained to reconstruct the masked values by
minimizing the mean squared

97 error (MSE) loss between the predicted and true expression values,
thereby updating model parameters

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.
>
> 98 (**Fig.** **1a**).
>
> 99 To accommodate the specific modality of bulk transcriptomic data,
> we designed a distinct model

100 architecture different from existing scLLMs. For each input sample,
we utilized ESM2, a large-scale

101 pretrained protein language model, to extract sequence-derived
embeddings of the canonical protein

102 products. These embeddings were used as the initial representations
for individual gene tokens, based

103 on the rationale that proteins, as the primary functional products
of genes, play central roles in biological

104 processes and directly reflect gene function at the molecular
level**20**. The rank of gene expression values

105 can be regarded as a form of gene ordering within each sample,
analogous to positional encoding in

106 LLMs. However, rank-based values are evenly spaced and fail to
capture the relative magnitudes

107 between gene expression levels. Inspired by rotary position encoding
(ROPE)**21**, we propose a rotary

108 expression embedding (REE) strategy to encode gene expression values
as positional representations,

109 preserving both their magnitude and continuity (**See** **Methods**
**for** **details**). The embeddings generated

110 by REE retain the relative relationships between gene expression
levels without requiring additional

111 training, offering strong stability and interpretability (**Fig.**
**1b**). In parallel, a multilayer perceptron (MLP)

112 module was employed to compress the input expression vector into a
global sample-level embedding.

113 These three representations were then integrated via element-wise
summation to form the final input

114 representations. The core architecture of BulkFormer consists of
stacked BulkFormer blocks, each

115 composed of one graph convolutional network (GCN)**22** layer
followed by *K* Performer**23** layers. The

116 GCN module leverages a prior biological knowledge graph to capture
explicit gene-gene relationships,

117 while the performer, a scalable variant of the transformer**24** is
employed to model implicit gene

118 interactions. The Performer approximates self-attention with linear
complexity, making it particularly

119 well-suited for processing high-dimensional bulk transcriptomic
inputs. After passing through *N*

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

120 BulkFormer blocks, the model yields contextualized gene embeddings,
which are projected through a

121 linear layer to generate scalar gene expression predictions
**(Fig.** **1c)**. Detailed hyperparameter settings

122 for BulkFormer, as well as results fromablation studies,
wereprovided inthe supplementaryinformation.

123 To assess BulkFormer’s ability, we evaluated its performance across
a series of canonical bulk

124 transcriptome tasks, including transcriptome imputation, disease
annotation, prognosis modeling, drug

125 response prediction, compound perturbation prediction, and gene
essentiality prediction **(Fig.** **1d)**. In all

126 tasks, BulkFormer consistently outperformed existing baseline
models, highlighting its capacity and

127 versatility for bulk transcriptome modeling (**Supplementary**
**Table** **1**). Notably, pretraining BulkFormer

128 on bulk transcriptomic data is an efficient and computationally
economical approach. BulkFormer

129 requires only 1% to 10% of the training time (measured by the
average single-GPU epoch duration)

130 compared to existing scLLMs, while still effectively capture
gene–gene relationships and generate high-

131 qualitygene-levelandtranscriptome-level representations
fordownstreamtasks (**Supplementary** **Table**

132 **2**).

133 **Pretraining** **on** **large-scale** **bulk** **transcriptomes**
**enables** **biologically** **meaningful** **embeddings**

134 Owing to technical limitations such as low mRNA capture efficiency
and sequencing depth per cell,

135 scRNA-seq typically detects only 500 to 5,000 genes per cell. By
comparison, bulk RNA-seq routinely

136 quantifies the expression of over 15,000 to 20,000 protein-coding
genes per sample, offering a more

137 complete view of the transcriptome. In bulk RNA-seq, the expression
level of each gene represents an

138 aggregate signal, reflecting the average expression across all
constituent cells within the sampled tissue

139 (**Fig.** **2a**). We further compared the sparsity of PreBULK with
that of the single-cell dataset sourced

140 from Tabula Sapiens**25** by evaluating the number of non-zero
protein-coding genes detected per gene

141 expression profile. On average, PreBULK captured expression for
16,606 protein-coding genes per

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

142 sample, whereas the single-cell dataset detected only 3,050 genes
per cell (**Fig.** **2b**). This stark contrast

143 underscores the difference in data sparsity between the two
modalities, highlighting the substantially

144 more complete transcriptomic coverage provided by bulk RNA-seq.
PreBULK encompasses bulk

145 transcriptomic profiles from nine major human physiological systems
and their associated tissues,

146 enabling BulkFormer to learn comprehensive representations of the
human bulk transcriptome (**Fig.** **2c**).

147 During the pretraining stage, BulkFormer was trained for a total of
29 epochs, with the MSE loss on the

148 test set steadily decreasing from an initial value of 7.5582 to
0.2427, indicating convergence and

149 completion of model training (**Fig.** **2d**). To visualize the
learned representations, we extracted the gene

150 token embeddings from BulkFormer’s embedding layer and applied
t-distributed stochastic neighbor

151 embedding (t-SNE)**26** for dimensionality reduction. The resulting
two-dimensional map revealed that

152 genes with distinct average expression levels were grouped into
different clusters **(Fig.** **2e)**. To evaluate

153 the biological relevance of the learned embeddings, we performed
K-means clustering on all gene

154 embeddings, partitioning them into 10 clusters **(Fig.** **2f)**.
Gene Ontology (GO)**27** enrichment analysis of

155 each cluster revealed a varying number of enriched terms, ranging
from 69 to 1,388 across clusters **(Fig.**

156 **2g)**, with distinct biological functions observed in different
gene groups **(Supplementary** **Fig.** **1)**. For

157 example, cluster 6 genes were predominantly associated with DNA
replication and cell cycle, whereas

158 cluster 9 genes were enriched in immune-related functions **(Fig.**
**2h)**. Consistently, Kyoto Encyclopedia

159 of Genes and Genomes (KEGG)**28** pathway enrichment analysis showed
that the number of enriched

160 pathways per cluster ranged from 12 to 172 **(Fig.** **2i)**, and
the functional annotations were likewise

161 cluster-specific **(Supplementary** **Fig.** **2)**. For instance,
cluster 2 genes were enriched in innate immune

162 signaling pathways, while cluster 7 genes were enriched in
cancer-related pathways **(Fig.** **2j)**.

163 Together, these results demonstrate that BulkFormer not only
successfully converges during pretraining

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

164 but also produces biologically meaningful gene embeddings that
reflect functional diversity within the

165 human transcriptome.

166 **BulkFormer** **enables** **context-aware** **imputation** **of**
**missing** **values** **from** **bulk** **transcriptomes**

167 BulkFormer was pretrained using a MLM strategy, in which a subset of
gene expression values is

168 masked and subsequently reconstructed based on contextual
information fromthe remaining genes. This

169 naturally lends BulkFormer to the task of imputing missing values in
bulk transcriptomic datasets. To

170 evaluate this capability, we randomly masked 15% of the gene
expression values in each sample within

171 an independent test set to simulate realistic dropout events, and
compared the imputation performance

172 of BulkFormer against a range of scLLMs. Geneformer, GeneCompass,
and scGPT were not directly

173 pretrained to model gene expression values, and therefore cannot
perform transcriptome imputation

174 tasks. We therefore included the variational autoencoder (VAE)**29**
as an additional baseline model for

175 comparison. As a result, BulkFormer achieved the highest imputation
accuracy, yielding a Pearson

176 correlation coefficient (PCC) of r = 0.954 between imputed and true
values **(Fig.** **3a)**. The next-best

177 model was the VAE (r = 0.806), followed by simple imputation methods
using the gene-wise mean (r =

178 0.754) or median (r = 0.743) across the training set. Models
pretrained on scRNA-seq data, including

179 scFoundation and scLong, performed substantially worse **(**r \<
0.150), likely due to their incompatibility

180 with the bulk data modality.

181 As BulkFormer imputes masked values based on contextualized
expression patterns, its performance is

182 expected to depend on the amount of contextual information
available. To test this hypothesis, we

183 evaluated imputation performance across a range of masking ratios on
the test set. Consistent with

184 expectations, performance peaked when the test-time masking ratio
matched the pretraining condition

185 (15%; r = 0.949) and gradually declined as the masking ratio
increased (**Fig.** **3b**). Notably, when the

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

186 masking ratio reached 35%, the performance dropped to r = 0.752,
comparable to simple mean- or

187 median-based imputation. These findings suggest that BulkFormer is
particularly advantageous when

188 the proportion of missing genes is below 35%, while simpler
statistical imputation methods may suffice

189 in cases of more extreme sparsity. To further assess its
generalizability, we evaluated BulkFormer on an

190 external test set comprising bulk RNA-seq profiles of 1,000 cancer
patients randomly selected from the

191 cancer genome atlas (TCGA) dataset. The model maintained strong
imputation performance, achieving

192 a PCC of r = 0.914 (**Fig.** **3c**), demonstrating its robustness
across diverse biological contexts.

193 Pancreatic cancer is among the deadliest malignancies, often
referred to as the “king of cancers,” and

194 poses a serious threat to human health**30**. Bulk RNA-seq enables
the identification of differentially

195 expressed genes (DEGs) between tumor and adjacent normal tissues in
patients with pancreatic cancer,

196 offering insights intotumorigenic mechanisms andopportunities for
therapeutic development. However,

197 due to technical limitations, the expression of certain potential
cancer-driver genes may be missing

198 during sequencing, thereby obscuring critical biological signals. In
such cases, BulkFormer’s

199 contextualized imputation capabilities can be leveraged to recover
missing gene expression values and

200 enhance transcriptomic completeness **(Fig.** **3d)**.To test this
application, we retrieved a publicly available

201 pancreatic cancer bulk RNA-seq dataset from the GEO database (GEO
accession number: GSE132956)

202 and used the limma R package**31** to identify DEGs between tumor
and normal samples, denoted as DEG1.

203 We then applied BulkFormer to impute missing gene expression values
across all samples and repeated

204 DEG analysis using the same pipeline to obtain a second set of DEGs,
denoted as DEG2. By computing

205 the set difference between DEG2 and DEG1, we identified 408
additional DEGs that emerged only after

206 imputation. Gene set enrichment analysis (GSEA)**32** of these 408
newly uncovered DEGs revealed

207 significant enrichment in biological processes such as cellular
respiration**,** oxidative phosphorylation**,**

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

208 and ATP metabolic process **(Fig.** **3e).** Notably**,** oxidative
phosphorylation emerged as one of the most

209 significantly enriched pathways, suggesting a potential role in
pancreatic tumor biology (**Fig.** **3f**).

210 Although cancer cells in hypoxic tumor microenvironments typically
rely on anaerobic glycolysis,

211 oxidative phosphorylation has been reported to be upregulated in
pancreatic ductal adenocarcinoma, as

212 well as in leukemias and lymphomas. Targeting this pathway with
oxidative phosphorylation inhibitors

213 has thus been proposed as a novel therapeutic strategy**33**. These
results demonstrate that BulkFormer not

214 only improves transcriptome completeness through imputation, but
also enables the discovery of non-

215 canonical, and potentially overlooked, disease mechanisms and
therapeutic targets in cancer.

216 Next, we evaluatedthe utilityofBulkFormer indiscovering novel
prognostic biomarkersacross multiple

217 cancer types. We selected bulk RNA-seq data from eight cancer
cohorts in TCGA, including breast

218 invasive carcinoma (BRCA)**,** kidney renal clear cell carcinoma
(KIRC)**,** liver hepatocellular carcinoma

219 (LIHC)**,** lung adenocarcinoma (LUAD)**,** ovarian serous
cystadenocarcinoma (OV)**,** pancreatic

220 adenocarcinoma (PAAD)**,** skin cutaneous melanoma (SKCM), and
stomach adenocarcinoma (STAD).

221 The number of alive and dead patients in each cohort is shown in
**Fig.** **3g**. For each cancer type, we first

222 identified genes with extremely high missingness, defined as those
absent in more than 95% of patient

223 samples. Such genes are usually1excluded from downstream analyses,
as conventional imputation

224 methods (e.g., mean or median imputation) assign uniform values
across all patients, eliminating

225 potential prognostic signals. Leveraging BulkFormer’s contextualized
imputation capabilities, we

226 recovered the expression values of these highly missing genes.
Patients were then stratified into high-

227 and low-expression groups based on the median of the imputed
expression values, followed by cox

228 proportional hazards regression to identify statistically
significant prognostic markers. In all of the

229 cancer types, we identified previously unrecognized protective and
risk-associated genes that robustly

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

230 stratified patient survival (**Fig.** **3h,** **Fig.** **3i**).
Notably, in KIRC, H4C1 emerged as a strong risk factor:

231 patients with high H4C1 expression level exhibited a 5.2-fold higher
mortality rate than those with low

232 expression level. In PAAD**,** PDE6H was identified as a protective
factor: patients with high PDE6H

233 expression level had a 74% lower mortality rate (hazard ratio =
0.26) than those with low expression

234 level (**Fig.** **3h,** **Fig.** **3i**). These newly discovered
biomarkers have not been previously reported in the

235 literature, likely duetothe highrate of missingexpression data. By
recovering such data ina biologically

236 informed manner, BulkFormer facilitates the identification of
overlooked yet clinically meaningful

237 prognostic biomarkers, with important implications for precision
oncology and downstream clinical

238 investigations.

239 **BulkFormer** **enables** **accurate** **classification** **of**
**disease** **types** **and** **cancer** **subtypes** **from** **bulk**

240 **transcriptomes**

241 Disease classification based on transcriptomic profiles is a
critical application in clinical research and

242 precision medicine. To evaluate BulkFormer’s power for disease-type
annotation, we obtained disease-

243 associated bulk RNA-seq data for 23 major human diseases from the
DiSignAtlas**34** database

244 (**Supplementary** **Table** **3**). We compared BulkFormer with
five existing scLLMs as baselines:

245 Geneformer**,** GeneCompass**,** scGPT**,** scFoundation, and
scLong. Specifically, transcriptomic

246 embeddings were extracted using BulkFormer and each baseline model,
followed by dimensionality

247 reduction using principal component analysis (PCA) to ensure
comparable feature spaces. A random

248 forest classifier was then trained for disease classification, and
model performance was evaluated using

249 the weighted F1 score. Ten-fold cross-validation showed that
BulkFormer achieved the highest overall

250 classification performance (weighted F1 = 0.939), followed by scGPT
(weighted F1 = 0.885) (**Fig.** **4a**).

251 We further evaluated model performance across individual disease
categories and found that

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

252 BulkFormer consistently outperformed all baselines across nearly all
disease types (**Fig.** **4b**),

253 highlighting its superior capability in disease annotation compared
to existing scLLMs.

254 We next assessed the ability of BulkFormer to classify cancer
subtypes. Bulk RNA-seq data from 33

255 cancer types were obtained from TCGA (**Supplementary** **Table**
**4**), and the same pipeline as in the

256 disease classification task was applied. In this more fine-grained
classification task, BulkFormer again

257 achieved the best performance (weighted F1 = 0.833),closely followed
by scGPT(weighted F1 = 0.830),

258 while the remaining baseline models performed substantially worse
(**Fig.** **4c**). To explore the

259 representational quality of BulkFormer embeddings, we visualized the
untrained transcriptomic

260 representations of patient samples using uniform manifold
approximation and projection (UMAP)**35**.

261 The resulting 2D projections revealed clear clustering by cancer
type, with well-separated boundaries

262 between different cancer classes (**Fig.** **4d**). We further
compared classification performance across each

263 individual cancer type, and observed that BulkFormer and scGPT
consistently delivered top-tier

264 performance across nearly all cancer types (**Supplementary**
**Fig.** **3**). Together, these results demonstrate

265 that BulkFormer not only enables accurate classification of broad
disease categories, but also performs

266 well in more granular subtype classification tasks, such as
distinguishing between diverse cancer types.

267 **BulkFormer** **enhances** **prognosis** **prediction** **by**
**generating** **context-aware** **gene** **representations**

268 It is an important task to predict the clinical outcomes of cancer
patients (survival or death), which are

269 known to be closely linked to their transcriptomic profiles, with
the expression levels of certain key

270 genes serving as valuable prognostic biomarkers**36**. To evaluate
whether BulkFormer can effectively

271 capture such prognostic signals, we first compared its performance
with that of baseline models in

272 predicting patient outcomes based on bulk RNA-seq data.

273 We collected transcriptomic profiles and survival outcome
information for ~10,000 patients across 33

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

274 cancer types from the TCGA database. The ratio of alive to dead
patients varied substantially across

275 different cancers (**Fig.** **5a**). For each model, we extracted
sample-level transcriptomic embeddings,

276 applied PCA to project them into a unified feature space, and then
trained a random forest classifier to

277 predict patient survival status. Ten-fold cross-validation revealed
that BulkFormer achieved the best

278 performance, with an AUROC of 0.747 and an AUPRC of 0.549.The
next-best performer, scFoundation,

279 yielded an AUROC of 0.726 and an AUPRC of 0.520 (**Fig.** **5b,**
**5c**).

280 While BulkFormer outperformed all baselines, there remains
considerable room for improvement. Akey

281 challenge in prognosis modeling lies in the high level of noise in
bulk transcriptomic data, because many

282 genes may be unrelated to patient outcomes. To mitigate this,
conventional approaches often rely on

283 univariate cox regression to identify prognostically relevant genes,
and subsequently combine their

284 expression values to compute risk scores. However, this strategy may
overlook weakly expressed or

285 subtly variable genes that nonetheless carry latent prognostic
information. To address this limitation, we

286 leveraged BulkFormer’s context-aware gene embeddings to amplify the
predictive power of otherwise

287 insignificant genes. Specifically, we adopted a supervised learning
framework inwhichBulkFormer was

288 used to extract contextualized gene embeddings for each patient
sample. Arandom forest classifier was

289 then trained to predict survival probability using these embeddings.
Using 10-fold cross-validation, we

290 obtained model-derived prediction scores for each patient, which
were subsequently used to stratify

291 patients into high- and low-risk groups based on the median score.
Cox and log-rank tests were then

292 applied to assessthe prognostic value of thesenew model-derived
scores. We applied this strategy across

293 eight representative cancer types and demonstrated enhanced
prognostic resolution for genes that were

294 initially non-significant. Remarkably, RPS27**,** a gene that
originally showed no prognostic value in

295 lower-grade glioma (LGG; HR = 1.0), became a highly significant risk
factor aftercontextual embedding

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

296 by BulkFormer (HR = 4.77) (**Fig.** **5d,** **5e**). These results
suggest that BulkFormer-generated gene

297 embeddings can rescue weak or overlooked biomarkers and enhance
their prognostic utility, providing

298 new opportunities for clinical research and precision medicine.

299 **BulkFormer** **facilitates** ***in*** ***silico*** **drug**
**discovery** **by** **modeling** **compound–transcriptome**

300 **relationships**

301 Transcriptomic profiles are essential for *in* *silico* drug
discovery, serving as both representations of

302 cellular identity and as response signatures following drug
perturbation. Compounds with similar

303 mechanisms of action typically induce concordant transcriptomic
changes, whereas pharmacologically

304 distinct agents elicit divergent expression profiles. Identifying
compounds that induce transcriptomic

305 responses inversely correlated with disease-associated signatures
enables phenotype-based drug

306 screening. Furthermore, functional enrichment of drug-induced gene
expression changes can provide

307 mechanistic insight into compound activity**37**. PRnet is a deep
learning framework designed to predict

308 transcriptomic responses to compound perturbations, trained on data
from the LINCS**38** database. The

309 LINCS database provides a large-scale resource of gene expression
profiles from various compounds

310 and cell lines under perturbation conditions. PRnet employs a
denoising autoencoder to integrate

311 untreated transcriptomic features with compound representations,
which are then decoded into predicted

312 post-treatment expression profiles**39**. However, PRnet reduces raw
gene expression profiles via linear

313 layers before integration, treating the transcriptome at a coarse
resolution and overlooking gene-specific

314 drug effects. To address this limitation, we incorporated
context-aware gene embeddings produced by

315 BulkFormer and scLLMs, fusing them with compound features at the
gene level for more fine-grained

316 representation. These fused features were passed through an MLP to
predict the post-treatment

317 expression levels of individual genes. We used the same dataset and
data-splitting protocol as PRnet and

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

318 evaluated model performance using PCC and spearman correlation
coefficients (SCC) between

319 predicted and ground-truth transcriptomic responses. Our results
demonstrate that models using gene-

320 level fusion consistently outperformed PRnet, indicating the
effectiveness of this strategy. Notably, the

321 model trained with BulkFormer-generated gene embeddings achieved the
best performance (PCC =

322 0.493; SCC = 0.430), substantially outperforming PRnet (PCC = 0.408;
SCC = 0.345) and scLLM-based

323 baselines (**Fig.** **6a,** **6b**). To assess the biological
relevance of the predicted gene expression responses,

324 we evaluated the case of Dovitinib **(Fig.** **6c**), an orally
bioavailable, potent inhibitor of class III–V

325 receptor tyrosine kinases, which was not present in the training
set. BulkFormer accurately predicted

326 the Dovitinib-induced transcriptomic changes, which were
subsequently analyzed via GSEA using the

327 KEGG pathway database. The results revealed significant
downregulation of cancer-related pathways,

328 including colorectal cancer, pancreatic cancer, and breast cancer
(**Fig.** **6d,** **6e**). Consistent with these

329 enrichment results, previous studies have shown that Dovitinib
combined with oxaliplatin exhibits

330 enhanced in vitro cytotoxicity in colon cancer cell lines,
regardless of RAS-RAF status**40**.

331 Drug response prediction is another crucial component of *in*
*silico* drug discovery, aiding in the

332 identification of compounds with selective sensitivity or resistance
across cancer types**41**. To evaluate

333 this task, we evaluated BulkFormer and scLLMs on a large-scale drug
response dataset curated from the

334 Genomics of Drug Sensitivity in Cancer (GDSC)**42** database, which
includes IC50 values for 255

335 compounds across 700 cancer cell lines representing 32 cancer types.
Compound features wereextracted

336 using the pretrained compound language model KPGT**43**, while
transcriptomic representations of cell

337 lines were derived from BulkFormer and baseline models. The features
were concatenated and passed

338 through an MLP to predict IC50 values. Ten-fold cross-validation
demonstrated that BulkFormer again

339 achieved the best performance (PCC = 0.910; SCC = 0.879), followed
by scFoundation (PCC = 0.880;

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

340 SCC = 0.848) (**Fig.** **6f,** **6g**), confirming its utility for
predictive modeling of drug sensitivity.

341 **BulkFormer** **improves** **gene** **essentiality** **prediction**
**based** **on** **bulk** **transcriptomic** **profiles**

342 Gene essentiality refers to the extent to which a gene is critical
for the survival and development of an

343 organism. Compared to non-essential genes, disruption of essential
genes often leads to severe

344 consequences, including lethality**44**. In cancer cells, gene
essentiality scores are typically defined as the

345 impact of gene knockout on cellular viability. Thus, targeting
essential genes that sustain cancer cell

346 growth represents a promising therapeutic strategy**45**. Gene
essentiality is influenced by various

347 biological factors, with gene expression levels being among the most
important. However, existing

348 methods rarely enable direct prediction of gene essentiality from
expression profiles alone. To address

349 this gap, we compiled a dataset from the DepMap**46** database,
consisting of gene expression levels and

350 gene essentiality scores for 17,862 protein-coding genes across
1,103 human cancer cell lines. We then

351 applied BulkFormer and scLLMs to generate context-aware gene
embeddings from the gene expression

352 profiles. These embeddings were used as input to an MLPto predict
the essentiality score for each gene.

353 Ten-fold cross-validation results showed that BulkFormer
significantly outperformed all baseline

354 models, achieving a PCC of 0.931 and a SCC of 0.759
(**Supplementary** **Fig.** **4**). These results

355 demonstrate that BulkFormer can more accurately infer the
essentiality of individual genes based solely

356 on transcriptomic data, enabling rapid identification of
cancer-specific vulnerabilities from patient-

357 derived expression profiles. This capability has important
implications for precision oncology, offering

358 a potential strategy to prioritize therapeutic targets for cancer
treatment.

359 **Discussion**

360 Recent advances in scLLMs have significantly improved the modeling
of sparse scRNA-seq data and

361 achieved state-of-the-art performance across a range of single-cell
tasks. It is known that scRNA-seq

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

362 typically detects only ~3000 genes per cell, while bulk RNA-seq can
profiling ~16,000 genes per sample.

363 Due to this fundamental difference in data modality, however, scLLMs
pretrained on sparse single-cell

364 data often underperform on clinical and tissue-level tasks that rely
on bulk RNA-seq. This limitation

365 motivated the development of a dedicated foundation model
specifically designed for bulk

366 transcriptomic data.

367 In this study, we developed BulkFormer, a large-scale foundation
model specifically designed for bulk-

368 level transcriptomic modeling.BulkFormer employs a hybrid encoder
architecturethat integrates a GNN

369 to capture explicit gene–gene relationships with a performer module
to learn implicit expression

370 dependencies. This structure enables the model to capture both
biological priors and context-dependent

371 gene interactions. BulkFormer was pretrained on over 500,000 bulk
transcriptomic profiles,

372 encompassing the expression of approximately 20,000 protein-coding
genes, allowing it to learn

373 comprehensive and biologically meaningful representations.

374 Across a series of bulk transcriptome-related downstream tasks,
including transcriptome imputation,

375 disease annotation, prognosis modeling, drug response prediction,
compound perturbation prediction,

376 and gene essentiality prediction, BulkFormer consistently
outperformed existing baseline and scRNA-

377 seq foundation models while incurring substantially lower training
costs. Notably, its high-quality

378 context-aware gene embeddings and superior imputation performance
enabled the discovery of

379 previously unrecognized prognostic biomarkers and the enhancement of
weakly predictive known

380 markers, providing valuable insights for cancer prognosis and
biomarker development.

381 Despite its strong performance, BulkFormer still face some
limitations. Its performance on single-cell

382 specific tasks such as cell type annotation is limited compared to
scLLMs because it was pretrained on

383 bulk RNA-seq data. This discrepancy reflects the intrinsic modality
gap between bulk and single-cell

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

384 transcriptomics. As such, we recommend that users select foundation
models based on their purpose to

385 ensure optimal performance. In addition, BulkFormer focuses
exclusively on protein-coding genes, and

386 does not include non-coding transcripts, which restricts its utility
for applications involving non-coding

387 RNA biology.

388 In the future, BulkFormer opens several avenues for future research.
First, the development of a unified

389 RNA-seq foundation model capable of jointly modeling bulk and
single-cell data could bridge modality

390 gaps and leverage the complementary strengths of both platforms.
Second, expanding the model to

391 include non-coding genes may provide new opportunities for studying
regulatory RNAbiology. Finally,

392 current transcriptome foundation models, including BulkFormer,
primarily rely on gene expression

393 matrices derived from RNA-seq data, while largely overlooking
sample-level meta-information. Such

394 metadata, including variables like sex, age, disease type, and
tissue of origin, may contain critical

395 covariates that influence gene expression patterns. Effectively
integrating these metadata with

396 transcriptomic profiles represents an important future direction in
the development of next-generation

397 transcriptome foundation models.

398 **Methods**

399 **Construction** **of** **the** **PreBULK** **dataset**

400 **Data** **collection**

401 Unlike scRNA-seq data, there is a notable lack of centralized public
resources that systematically curate

402 bulk RNA-seq datasets. To address this, we manually collected human
bulk RNA-seq profiles from

403 public repositories including the Gene Expression Omnibus (GEO) and
ARCHS4. Duplicate entries

404 were removed based on GEO sample identifiers. To ensure data
consistency and compatibility for model

405 pretraining, only samples with available raw count matrices were
retained. The resulting dataset spans

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

406 nine major human physiological systems and includes samples from
both healthy and diseased states.

407 **Gene** **ID** **unification**

408 To ensure consistency across datasets, all gene expression matrices
were unified using Ensembl gene

409 identifiers, as provided by the Ensembl**47** database. We included
all annotated human protein-coding

410 genes, totaling 20,010 genes. For samples in which specific gene
entries were absent, zero-padding was

411 applied to maintain a consistent dimensionality across all
expression profiles.

412 **Quality** **control**

413 To remove low-quality samples, we filtered the dataset using the
NumPy package by excluding

414 expression profiles with fewer than 14,000 non-zero gene values.
This threshold effectively eliminates

415 potentially misclassified or contaminated scRNA-seq data, reduces
sparsity, and ensures the retained

416 samples reflect true bulk transcriptomic profiles. Finally, the
constructed PreBULK dataset comprises a

417 total of 522,769 bulk transcriptomic samples.

418 **Data** **preprocessing**

419 To correct for differences in sequencing depth and gene length, and
to scale raw count values to a

420 comparable range, we converted raw gene expression counts to
transcripts per million (TPM). The TPM

421 value for each gene was calculated as follows:

> 𝐶

422 𝑇𝑃𝑀 = 𝐿𝑖 × 106

> 𝑁
>
> 𝑗=1 𝐿

423 Where 𝐶 is the raw count of gene 𝑖, 𝐿𝑖 is the length of gene 𝑖 and 𝑁
is the total number of genes.

424 This normalization ensures that gene expression levels are
comparable both within and across samples.

425 **BulkFormer** **model** **architecture**

426 **Embedding** **module**

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

427 The core principle of LLM-like models is to represent tokens as
high-dimensional embeddings, which

428 are optimized through pretraining such that semantically related
tokens acquire similar representations.

429 In the context of transcriptomic data, this involves two key
components: the representation of gene

430 identities and the encoding of gene expression values.

431 To represent gene identities, we retrieved the primary protein
product sequence of each gene from the

432 Ensembl database and extracted sequence-based embeddings using the
ESM2 model. These protein

433 embeddings were aggregated using mean pooling to generate the
initial gene embedding. Compared

434 with one-hot encoding, ESM2–based embeddings capture functional and
evolutionary relationships

435 among genes more effectively, thereby enhancing BulkFormer’s ability
to learn biologically meaningful

436 gene–gene dependencies.

437 Gene expression values within a transcriptome can be interpreted as
defining an internal ordering of

438 gene tokens, reflecting their relative biological activity. A naïve
approach would involve rank-based

439 encoding, but this method suffers from two major limitations. First,
it reduces data resolution by

440 discarding the absolute magnitude of expression values. Second, rank
values are sample-specific and

441 not directly comparable across samples. For example, two genes with
the same rank in different samples

442 may have substantially different expression magnitudes. To overcome
these limitations, we propose a

443 rotary expression embedding (REE) strategy inspired by rotary
position encoding (ROPE).While ROPE

444 is traditionally used to encode positional information in natural
language processing, we adapt this

445 concept to embed absolute gene expression values, preserving both
their magnitude and continuity.

446 Specifically, for gene 𝑔 with expression value 𝑥 , we defined the
expression embedding as:

447 𝐸𝑚𝑏𝑒𝑥𝑝(𝑥 ) = \[𝑠𝑖𝑛(𝛩𝑥 ),𝑐𝑜𝑠(𝛩𝑥 )\]

448 where the frequency matrix is defined as:

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

449 𝛩 = {𝜃 = 1002𝑖/𝑑\| 𝑖 ∈ \[1,2,...,𝑑/2\]}

450 and *d* is the embedding dimension. Prior to this transformation,
expression values are normalized using

451 𝑙𝑜𝑔(𝑇𝑃𝑀 + 1) to ensure numerical stability.

452 In addition, we incorporated sample-level context by applying an MLP
to compress the entire gene

453 expression vector of each sample into a low-dimensional embedding,
capturing its global transcriptomic

454 state.

455 Finally, the three types of embedding, including the gene identity
embedding 𝐸𝑆𝑀2, the expression

456 value embedding 𝐸 𝐸𝐸, and the sample context embedding 𝐸𝑀𝐿𝑃, are
combined through element-wise

457 addition to generate the final model input.

458 𝐼𝑛𝑝𝑢𝑡 = 𝐸𝑆𝑀2 + 𝑅𝐸𝐸 + 𝐸𝑀𝐿𝑃

459 **BulkFormer** **blocks**

460 The core architecture of BulkFormer consists of *N* stacked
BulkFormer blocks, each comprising a graph

461 convolutional network (GCN) layer followed by *K* layers of
performer-based self-attention. The GCN

462 layers aredesigned to captureexplicit gene–gene relationships
derived from biological prior knowledge,

463 while the Performer layers model global, implicit dependencies
across genes.

464 To incorporate prior biological knowledge, we first construct a
gene-generelationshipgraph 𝐺 = (𝑉,𝐸),

465 where 𝑉 is the set of genes and 𝐸 denotes the edges representing
gene-gene associations derived from

466 known gene knowledge graph. In the ablation studies, we
systematically evaluated the impact of

467 different knowledge graphs on model performance, including the gene
co-expression graph, the gene

468 ontology (GO) similarity graph, and the protein–protein interaction
(PPI) graph. Among these, the gene

469 co-expression graph achieved the best performance and was therefore
adopted in the GCN module of

470 BulkFormer (**Supplementary** **Fig.** **5**). We constructed the
gene co-expression graph using the absolute

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

471 values of Pearson correlation coefficients (PCC) between gene
expression profiles as edge weights. To

472 avoid excessive graph density and retain the most informative
interactions, we preserved only the top

473 20 edges withthe highest weights for eachnode. Additionally, edges
withPCC below 0.4 werediscarded

474 to eliminate weak or spurious gene–gene associations.

475 The GCN layers updates the gene embeddings based on the graph
structure as follows:

476 𝐻(𝑙+1) = 𝜎 (𝐷−1 𝐴 𝐷−1 𝐻(𝑙)𝑊(𝑙) )

477 where，𝐻(𝑙) denotes the feature matrix at the 𝑙 − 𝑡ℎ layer, 𝐴 = 𝐴 +
𝐼 is the adjacency matrix with

478 self-loops, 𝐷 is the corresponding degree matrix, 𝑊(𝑙) is a
learnable weight matrix, and 𝜎(∙) is a

479 nonlinear activation function.

480 Following the GCN layers, we employed Performer, a kernel-based
transformer variant, to capture long-

481 range gene-gene interactions in transcriptomic profiles \[ref\].
Unlike standard self-attention mechanisms,

482 Performer approximates attention scores via a kernelizable
formulation to achieve linear scalability.

483 Specifically, the attention computation is expressed as:

484 𝐴𝑡𝑡(𝑄,𝐾,𝑉) = 𝐷−1 (𝜙(𝑄)(𝜙(𝐾))𝑇𝑉)

485 𝐷−1 = 𝑑𝑖𝑎𝑔 (𝜙(𝑄)(𝜙(𝐾))𝑇1𝐾)

486 where 𝑄 = 𝑋𝑊, 𝐾 = 𝑋𝑊 and 𝑉 = 𝑋𝑊 are linear transformation of the
input *X*, 𝑊 are training

487 parameters, 𝜙(∙) is a kernel function that used for approximating
the exponential attention matrix, 1𝐾

488 is the all-ones vector oflength 𝐾, and 𝑑𝑖𝑎𝑔(∙) is a diagonal
matrixwiththe inputvector as the diagonal.

489 This formulation eliminates the quadratic dependency on sequence
length, enabling efficient modeling

490 of large-scale transcriptomic inputs while preserving key
interaction patterns.

491 After passing through the stacked BulkFormer blocks, the updated
gene representations are then fed into

492 an MLP to output the final gene expression values.

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

493 **BulkFormer** **pretraining** **task**

494 During the pretraining stage, BulkFormer was trained using a masked
language modeling (MLM)

495 strategy in a self-supervised manner. Specifically, approximately
15% of gene expression values in each

496 input transcriptome were randomly selected and masked using a
special placeholder token (−10),

497 resulting in a partially masked input vector. The model is then
tasked with reconstructing the masked

498 expression values based on the unmasked context. The training
objective is to minimize the

499 reconstruction error between the predicted and true expression
values at the masked positions, using the

500 mean squared error (MSE) as the loss function ℒ:

501 ℒ = \|𝛭\| ∑ (𝑦 − ̂ )2 𝑚∈𝛭

502 where 𝛭 denotes the set of masked gene indices, 𝑦 is the true
expression value at position 𝑚, and

503 ̂ is the model’s corresponding predicted value.

504 All parameters 𝛩 = {𝐸𝑛𝑝𝑢𝑡,𝛩𝐺𝐶𝑁,𝛩𝑃𝑒𝑟𝑓𝑜𝑟𝑚𝑒𝑟,𝛩𝑀𝐿𝑃} were optimized
during the pretraining stage.

505 The detailed hyperparameter setting of the BulkFormer can be found
in **Supplementary** **Table** **5**.

506 **BulkFormer** **implementation** **details**

507 BulkFormer was implemented in PyTorch (v2.5.1) with CUDA12.4 and
Python 3.12.7. Pretraining was

508 conducted on eight NVIDIAA800 GPUs for 29 epochs, requiring
approximately 350 GPU hours. We

509 used the AdamW optimizer with a max learning rate of 0.0001. The
learning rate was linearly warmed

510 up from zero, reaching its peak after 5% of total training steps. To
support large model training, we

511 adopted a per-device batch size of 4 and employed gradient
accumulation with 128 steps, resulting in a

512 large effective batch size.

513 **Downstream** **methods**

514 BulkFormer's downstream tasks fall into three major categories: (1)
Expression-level prediction tasks:

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

515 These involve imputing missing gene expression values in bulk
transcriptomes using context-aware

516 predictions directly output by BulkFormer. (2) Gene-level embedding
tasks: These leverage gene

517 embeddings produced by the final Performer layer of BulkFormer as
input features for downstream

518 applications such as compound perturbation prediction, gene
essentiality estimation, and prognostic

519 modeling of individual genes. (3) Sample-level embedding tasks: In
these tasks, gene embeddings from

520 the final Performer layer are aggregated via max pooling to obtain
sample-level representations. These

521 representations are then used for downstream analyses, including
disease classification, cancer subtype

522 annotation, drug response prediction, and patient prognosis
modeling.

523 We compared BulkFormer against five representative scLLMs, including
Geneformer, GeneCompass,

524 scGPT, scFoundation, and scLong. Due to differences in model
architectures and pretraining objectives,

525 we adopted task-specific evaluation strategies for fair comparison.
For expression-level prediction tasks,

526 only scFoundation and scLong were included as baselines, since they
are the only models pretrained to

527 reconstruct gene expression values directly. For gene-level
embedding tasks, we excluded scLong and

528 focused on the remaining scLLMs. As these models were pretrained
using only the top 1,000–2,000

529 highly expressed genes per cell rather than full-length
transcriptomes, we divided each bulk

530 transcriptome into ten non-overlapping blocks and sequentially input
them into the scLLMs to extract

531 gene-level embeddings. For sample-level embedding tasks, we used the
top 2,000 most highly expressed

532 genes from each bulk sample and applied max pooling over their
embeddings to generate sample-level

533 representations for downstream classification and regression tasks.

534 **Evaluation** **metrics**

535 We used several quantitative metrics to evaluate the performance of
BulkFormer and baseline models

536 across different downstream tasks.

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

537

538

539

540

541

542

543

544

545

546

547

548

549

550

551

552

553

554

555

556

557

**Pearson** **correlation** **coefficient** **(PCC)**

> P𝐶𝐶 = ∑𝑖=1(𝑥𝑖 − 𝑥)(𝑦 − 𝑦) √∑𝑖=1(𝑥𝑖 − 𝑥) √∑𝑖=1(𝑦 − 𝑦)

**Spearman** **correlation** **coefficient** **(SCC)**

> 𝑆𝐶𝐶 = 1 − 𝑛(𝑛𝑖 −1)

where 𝑑𝑖 is the difference between the ranks of corresponding variables.

**Area** **under** **the** **receiver** **operating** **characteristic**
**curve** **(AUROC)**

AUROC evaluates the ability of a model to distinguish between positive
and negative classes across

different decision thresholds. It is computed as the area under the ROC
curve.

**Area** **under** **the** **precision-recall** **curve** **(AUPRC)**

AUPRC summarizes the trade-off between precision and recall.
Particularly useful for imbalanced

datasets, it is calculated as the area under the precision-recall curve.

**F1** **score**

The F1 score is the harmonic mean of precision and recall, defined as:

> 2 ∙ 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 ∙ 𝑅𝑒𝑐𝑎𝑙𝑙 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 + 𝑅𝑒𝑐𝑎𝑙𝑙
>
> Precision = 𝑇𝑃+ 𝐹𝑃
>
> Recall = 𝑇𝑃 + 𝐹𝑁

where TP (True Positive) denotes the number of samples correctly
predicted as the positive class, FP

(False Positive) represents the number of negative samples incorrectly
predicted as positive, and FN

(False Negative) refers to the number of positive samples incorrectly
predicted as negative.

**Weighted** **F1** **score**

For multi-class settings, the weighted F1 is defined as:

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.
>
> 𝐶

558 Weighted F1 = ∑ 𝑤 ∙ F1𝑐 𝑐=1

559 𝑤 = 𝑛𝑐 𝑖=1 𝑖

560 Where 𝐶 is the number of classes, 𝑛𝑐 is the number of samples in
class 𝑐, and F1𝑐 is the F1 score

561 for class 𝑐.

562 **Data** **availability**

563 The large-scale human bulk transcriptomes were collected from the
GEO

564 (https://www.ncbi.nlm.nih.gov/geo/) and the ARCHS4
(https://archs4.org/) database. The

565 transcriptomic profiles and corresponding survival information of
cancer patients were downloaded

566 from the TCGA (https://portal.gdc.cancer.gov/) database. The data
used for the disease classification

567 task were obtained from the DiSignAtlas
(http://www.inbirg.com/disignatlas/home) database.

568 Compound perturbation data were obtained from the LINCS
(https://clue.io/) database. Drug response

569 data were accessed from the GDSC (https://www.cancerrxgene.org/)
database. Gene essentiality data

570 were downloaded from the DepMap (https://depmap.org/portal/)
database. The data used in this study

571 are available on Zenodo (https://doi.org/10.5281/zenodo.15559368).
Source data are provided with this

572 paper.

573 **Code** **availability**

574 BulkFormer source code is available on GitHub
(https://github.com/KangBoming/BulkFormer).

575 **Acknowledgements**

576 This study was supported by the grants from the National Natural
Science Foundation of China

577 \[62025102\].

578 **Author** **information**

579 These authors contributed equally: Boming Kang, Rui Fan.

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

580 **Authors** **and** **Affiliations**

581 **Department** **of** **Biomedical** **Informatics,** **State**
**Key** **Laboratory** **of** **Vascular** **Homeostasis** **and**

582 **Remodeling,** **School** **of** **Basic** **Medical**
**Sciences,** **Peking** **University,** **38** **Xueyuan** **Rd,**
**Beijing,** **100191,**

583 **China**

584 Boming Kang, Rui Fan & Qinghua Cui

585 **School** **of** **Pharmaceutical** **Sciences,** **Jilin**
**University,** **Changchun** **130021,** **China**

586 Meizheng Yi

587 **School** **of** **Sports** **Medicine,** **Wuhan** **Sports**
**University,** **No.** **461** **Luoyu** **Rd.** **Wuchang**
**District,**

588 **Wuhan** **430079,** **Hubei** **Province,** **China**

589 Chunmei Cui & Qinghua Cui

590 **Contributions**

591 BK and RF designed the study. BK, RF and MY performed the study. BK,
RF, MY, CC and QC wrote

592 or edited the manuscript. QC supervised the study.

593 **Corresponding** **author**

594 Correspondence to Qinghua Cui

595

596

597

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

598 **Reference**

599 1 Brown, T. *et* *al.* Language models are few-shot learners.
*Advances* *in* *neural* *information* 600 *processing* *systems*
**33**, 1877-1901 (2020).

601 2 Simon, E., Swanson, K. & Zou, J. Language models for biological
research: a primer. 602 *Nature* *Methods* **21**, 1422-1429 (2024).
<u>https://doi.org:10.1038/s41592-024-02354-y</u> 603 3 Ji, Y., Zhou,
Z., Liu, H. &Davuluri,R. V. DNABERT: pre-trained Bidirectional Encoder
604 Representations from Transformers model for DNA-language in genome.
605 *Bioinformatics* **37**, 2112-2120 (2021).

606 4 Chen, J. *et* *al.* Interpretable RNA foundation model from
unannotated data for highly 607 accurate RNA structure and function
predictions. *arXiv* *preprint* *arXiv:2204.00300* 608 (2022).

609 5 Lin, Z. *et* *al.* Evolutionary-scale prediction of atomic-level
protein structure with a 610 language model. *Science* **379**,
1123-1130 (2023).

611 6 Szałata, A. *et* *al.* Transformers in single-cell omics: a review
and new perspectives. 612 *Nature* *Methods* **21**, 1430-1443 (2024).
<u>https://doi.org:10.1038/s41592-024-02353-z</u> 613 7 Wu, S., Yin, C.,
Wang, Y. & Sun, H. Identifying cancer prognosis genes through causal 614
learning. *Briefings* *in* *Bioinformatics* **26**, bbae721 (2025).

615 8 Theodoris, C. V. *et* *al.* Transfer learning enables predictions
in network biology. *Nature* 616 **618**, 616-624 (2023).
<u>https://doi.org:10.1038/s41586-023-06139-9</u>

617 9 Cui, H. *et* *al.* scGPT: toward building a foundation model for
single-cell multi-omics 618 using generative AI. *Nature* *Methods*
**21**, 1470-1480 (2024). 619
<u>https://doi.org:10.1038/s41592-024-02201-0</u>

620 10 Hao, M. *et* *al.* Large-scale foundation model on single-cell
transcriptomics. *Nature* 621 *Methods* **21**, 1481-1491 (2024).
<u>https://doi.org:10.1038/s41592-024-02305-7</u>

622 11 Yang, X. *et* *al.* GeneCompass: deciphering universal gene
regulatory mechanisms with 623 a knowledge-informed cross-species
foundation model. *Cell* *Research* **34**, 830-845 624 (2024).
<u>https://doi.org:10.1038/s41422-024-01034-y</u>

625 12 Bai, D.*et* *al.*scLong: ABillion-Parameter Foundation Modelfor
Capturing Long-Range 626 Gene Contextin Single-Cell Transcriptomics.
*bioRxiv*, 2024.2011. 2009.622759 (2024). 627 13 Yang, F. *et* *al.*
scBERT as a large-scale pretrained deep language model for cell type 628
annotation of single-cell RNA-seq data.
*NatureMachineIntelligence***4**, 852-866 (2022). 629
<u>https://doi.org:10.1038/s42256-022-00534-z</u>

630 14 Li, C. *et* *al.* Benchmarking AI Models for In Silico Gene
Perturbation of Cells. (2025). 631
<u>https://doi.org:10.1101/2024.12.20.629581</u>

632 15 Li, T. *et* *al.* An overview of computational methods in
single-cell transcriptomic cell 633 type annotation. *Briefings* *in*
*Bioinformatics* **26**, bbaf207 (2025).

634 16 Park, S. & Lee, H. Robust self-supervised learning strategy to
tackle the inherent 635 sparsity in single-cell RNA-seq data.
*Briefings* *in* *Bioinformatics* **25**, bbae586 (2024). 636 17 Edgar,
R., Domrachev, M. & Lash, A. E. Gene Expression Omnibus: NCBI gene 637
expression and hybridization array data repository. *Nucleic* *acids*
*research* **30**, 207-210 638 (2002).

639 18 Lachmann, A. *et* *al.* Massive mining of publicly available
RNA-seq data from human 640 and mouse. *Nature* *communications* **9**,
1366 (2018).

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

641 19 Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. in
*Proceedings* *of* *the* *2019* 642 *conference* *of* *the* *North*
*American* *chapter* *of* *the* *association* *for* *computational* 643
*linguistics:* *human* *language* *technologies,* *volume* *1* *(long*
*and* *short* *papers).* 4171-644 4186.

645 20 Kulmanov, M. *et* *al.* Protein function prediction as
approximate semantic entailment. 646 *Nature* *Machine* *Intelligence*
**6**, 220-228 (2024).

647 21 Su, J. *et* *al.* Roformer: Enhanced transformer with rotary
position embedding. 648 *Neurocomputing* **568**, 127063 (2024).

649 22 Kipf, T. N. & Welling, M. Semi-supervised classification with
graph convolutional 650 networks. *arXiv* *preprint* *arXiv:1609.02907*
(2016).

651 23 Choromanski, K. *et* *al.* Rethinking attention with performers.
*arXiv* *preprint* 652 *arXiv:2009.14794* (2020).

653 24 Vaswani, A. *et* *al.*Attention is all you need. *Advances* *in*
*neural* *information* *processing* 654 *systems* **30** (2017).

655 25 Consortium\*, T. T. S. *et* *al.* The Tabula Sapiens: A
multiple-organ, single-cell 656 transcriptomic atlas of humans.
*Science* **376**, eabl4896 (2022).

657 26 Cieslak, M. C., Castelfranco, A. M., Roncalli, V., Lenz, P. H. &
Hartline, D. K. t-658 Distributed Stochastic Neighbor Embedding (t-SNE):
A tool for eco-physiological 659 transcriptomic analysis. *Marine*
*genomics* **51**, 100723 (2020).

660 27 Ashburner, M. *et* *al.* Gene ontology: tool for the unification
of biology. *Nature* *genetics* 661 **25**, 25-29 (2000).

662 28 Kanehisa, M. & Goto, S. KEGG: kyoto encyclopedia of genes and
genomes. *Nucleic* 663 *acids* *research* **28**, 27-30 (2000).

664 29 Kingma, D. P. & Welling, M. (Banff, Canada, 2013).

665 30 Hariharan, D., Saied, A. & Kocher, H. Analysis of mortality rates
for pancreatic cancer 666 across the world. *Hpb* **10**, 58-62 (2008).

667 31 Smyth, G. K. in *Bioinformatics* *and* *computational* *biology*
*solutions* *using* *R* *and* 668 *Bioconductor* 397-420 (Springer,
2005).

669 32 Subramanian, A. *et* *al.* Gene set enrichment analysis: a
knowledge-based approach for 670 interpreting genome-wide expression
profiles. *Proceedings* *of* *the* *National* *Academy* *of* 671
*Sciences* **102**, 15545-15550 (2005).

672 33 Ashton, T. M., McKenna, W. G., Kunz-Schughart, L. A. & Higgins,
G. S. Oxidative 673 phosphorylation as an emerging target in cancer
therapy. *Clinical* *Cancer* *Research* **24**, 674 2482-2490 (2018).

675 34 Zhai, Z. *et* *al.* DiSignAtlas: an atlas of human and mouse
disease signatures based on 676 bulk and single-cell transcriptomics.
*Nucleic* *Acids* *Research* **52**, D1236-D1245 (2024). 677 35 McInnes,
L., Healy, J. & Melville, J. Umap: Uniform manifold approximation and
678 projection for dimension reduction. *arXiv* *preprint*
*arXiv:1802.03426* (2018).

679 36 Nagy, Á., Munkácsy, G. & Győrffy, B. Pancancer survival analysis
of cancer hallmark 680 genes. *Scientific* *reports* **11**, 6047
(2021).

681 37 Lamb, J. *et* *al.* The Connectivity Map: using gene-expression
signatures to connect 682 small molecules, genes, and disease. *science*
**313**, 1929-1935 (2006).

683 38 Subramanian, A. *et* *al.*Anext generation connectivity map:
L1000 platform and the first 684 1,000,000 profiles. *Cell* **171**,
1437-1452. e1417 (2017).

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

685 39 Qi, X. *et* *al.* Predicting transcriptional responses to novel
chemical perturbations using 686 deep generative model for drug
discovery. *Nature* *Communications* **15**, 1-19 (2024). 687 40 Gaur,
S. *et* *al.* Dovitinib synergizes with oxaliplatin in suppressing cell
proliferation and 688 inducing apoptosis in colorectal cancer cells
regardless of RAS-RAF mutation status. 689 *Molecular* *Cancer* **13**,
1-16 (2014).

690 41 Tang, Y.-C. & Gottlieb, A. Explainable drug sensitivity
prediction through cancer 691 pathway enrichment. *Scientific* *reports*
**11**, 3128 (2021).

692 42 Yang, W. *et* *al.* Genomics of Drug Sensitivity in Cancer
(GDSC): a resource for 693 therapeutic biomarker discovery in cancer
cells. *Nucleic* *acids* *research* **41**, D955-D961 694 (2012).

695 43 Li, H. *et* *al.* A knowledge-guided pre-training framework for
improving molecular 696 representation learning. *Nature*
*Communications* **14**, 7568 (2023).

697 44 Bartha, I., Di Iulio, J., Venter, J. C. & Telenti, A. Human gene
essentiality. *Nature* 698 *Reviews* *Genetics* **19**, 51-62 (2018).

699 45 Kang, B., Fan, R., Cui, C. & Cui, Q. Comprehensive prediction and
analysis of human 700 protein essentiality based on a pretrained large
language model. *Nature* *Computational* 701 *Science* **5**, 196-206
(2025).

702 46 Tsherniak, A. *et* *al.* Defining a cancer dependency map. *Cell*
**170**, 564-576. e516 (2017). 703 47 Harrison, P. W. *et* *al.* Ensembl
2024. *Nucleic* *acids* *research* **52**, D891-D899 (2024). 704

705

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

706 **Figure** **Legends**

707 **Fig.** **1.** **Overview** **of** **the** **BulkFormer**
**framework** **and** **its** **applications**. **(a)** The pretraining
phase of

708 BulkFormer adopts a masked language modeling (MLM) strategy, in
which approximately 15% of gene

709 expression values in each input sample are randomly masked. The
model is trained to predict the masked

710 values based on context,and parameters are optimized using the mean
squared error (MSE) loss between

711 the predicted and the true values. **(b)** Schematic illustration of
the rotary expression embedding (REE)

712 strategy for encoding gene expression values. **(c)** Model
architecture of BulkFormer. ESM2 was used to

713 extract sequence-based embeddings of canonical protein products,
serving as initial representations for

714 individual gene tokens. Each gene’s expression value was treated as
a positional token and encoded

715 using rotary position embedding to capture continuous expression
information. Simultaneously, a MLP

716 module compressed the global expression vector into a sample-level
embedding. These three

717 representations were fused via element-wise summation to form the
final model input. The core of

718 BulkFormer consists of stacked blocks, each containing a graph
convolutional network layer to model

719 gene–gene relationships followed by *K* Performer layers to capture
long-range interactions.After *N* such

720 blocks, contextualized gene embeddings are output and passed through
a linear projection layer to

721 predict gene expression levels. **(d)** Downstream applications of
BulkFormer include transcriptome

722 imputation, disease annotation, prognosis modeling, drug response
prediction, compound perturbation

723 prediction, and gene essentiality prediction.

724 **Fig.** **2.** **BulkFormerpretrained** **on** **large-scale**
**bulk** **transcriptomes** **enables** **biologically** **meaningful**

725 **embeddings**. **(a)** Schematic illustration comparing the
modality characteristics of single-cell RNA-seq

726 (scRNA-seq) and bulk RNA-seq. **(b)** Comparison of the number of
detected genes per sample between

727 BulkFormer’s pretraining bulk RNA-seq dataset and the single-cell
transcriptomes from the Tabula

728 Sapiens database. **(c)**Distribution ofhuman
organsystemsrepresentedinthe bulktranscriptomic dataset

729 used for BulkFormer pretraining. **(d)** Loss curve on the held-out
test set during BulkFormer pretraining.

730 **(e)** t-SNE visualization of gene embeddings extracted from
BulkFormer’s embedding layer. Color

731 intensity reflects the average expression level (TPM: transcripts
per million) of each gene across the

732 training set. **(f)** K-means clustering of gene embeddings (k =
10), followed by t-SNE dimensionality

733 reduction for visualization. **(g)** Number of significantly
enriched Gene Ontology (GO) terms associated

734 with genes in each cluster. **(h)** Comparative GO term enrichment
results for Gene Cluster 6 and Gene

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

735 Cluster 9. **(i)** Number of significantly enriched KEGG pathways
associated with genes in each cluster.

736 **(j)** Comparative KEGG term enrichment results for Gene Cluster 2
and Gene Cluster 7.

737 **Fig.** **3.BulkFormerenables** **context-aware** **imputation**
**ofmissing** **values** **frombulk** **transcriptomes.**

738 **(a)** Performance comparison between BulkFormer and baseline
models on the transcriptome imputation

739 task. **(b)** Effect of varying gene expression missing rates on the
imputation performance of BulkFormer.

740 **(c)** The imputation results of BulkFormer on transcriptomic data
from the TCGA database. **(d)**

741 Schematic illustration of BulkFormer’s ability to contextually
impute gene expression values even for

742 genes that are completely missing from a sample. **(e–f)** GSEA
results of differentially expressed genes

743 newly identified after BulkFormer-based imputation. **(g)**
Distribution of survival and death outcomes in

744 eight selected cancer patient cohorts from the TCGA database.
**(h)** Prognostic biomarkers newly

745 discovered via BulkFormer-based imputation, including both risk and
protective factors. Genes with

746 HR \> 1 are defined as risk factors, and those with HR \< 1 are
defined as protective factors. **(i)** Kaplan-

747 Meier survival curves for cancer patients stratified by expression
levels of newly discovered prognostic

748 biomarkers imputed by BulkFormer. PCC: Pearson correlation
coefficient. SCC: Spearman correlation

749 coefficient. NES: normalized enrichment score. HR: hazard ratio.
Statistical tests: The prognostic

750 markers shown in (h) were identified using univariate Cox regression
analysis. Survival curves in (i)

751 were compared using the log-rank test. Significance levels are
indicated as follows: \*p \< 0.05, \*\*p \<

752 0.01, \*\*\*p \< 0.001.

753 **Fig.** **4.** **BulkFormer** **enables** **accurate**
**classification** **of** **disease** **types** **and** **cancer**
**subtypes** **from** **bulk**

754 **transcriptomes.** **(a)** Overall performance comparison of
BulkFormer and baseline models on the

755 disease classification task. (b) Per-disease classification
performance of BulkFormer and other baseline

756 models. (c) Performance comparison on cancer subtype classification
across different models. (d)

757 UMAP visualization of sample-level embeddings extracted by
BulkFormer for transcriptomes from

758 different cancer types.

759 **Fig.** **5.** **BulkFormer** **enhances** **prognosis**
**prediction** **by** **generating** **context-aware** **gene**

760 **representations.** **(a)** Distribution of survival outcomes
(alive vs. dead) across 33 cancer cohorts in the

761 TCGA database. **(b-c)** Performance comparison of BulkFormer and
baseline models in predicting

762 patient prognosis. **(d)** BulkFormer-enhanced biomarkers for
prognosis prediction. **(e)** Kaplan-Meier

763 survival curves based on BulkFormer-enhanced prognostic biomarkers.
AUROC: area under the

764 receiver operating characteristic curve. AUPRC: area under the
precision-recall curve. HR: hazard ratio.

> bioRxiv preprint doi:
> [https://doi.org/10.1101/2025.06.11.659222;](https://doi.org/10.1101/2025.06.11.659222)
> this version posted June 17, 2025. The copyright holder has placed
> this preprint (which was not certified by peer review) in the Public
> Domain. It is no longer restricted by copyright. Anyone can legally
> share, reuse, remix, or adapt this material for any purpose without
> crediting the original authors.

765 Statistical tests: The prognostic markers shown in (d) were
identified using univariate Cox regression

766 analysis. Survival curves in (e) were compared using the log-rank
test. Significance levels are indicated

767 as follows: \*p \< 0.05, \*\*p \< 0.01, \*\*\*p \< 0.001.

768 **Fig.** **6.** **BulkFormer** **facilitates** ***in*** ***silico***
**drug** **discovery** **by** **modeling** **compound–transcriptome**

769 **relationships.** **(a-b)** Performance comparison of BulkFormer
and baseline models on the compound

770 perturbation prediction task. **(c)** Chemical structure of the
compound Dovitinib. **(d-e)** GSEAenrichment

771 analysis results based on differentially expressed genes predicted
by BulkFormer following Dovitinib

772 perturbation. **(f-g)** Performance comparison of BulkFormer and
baseline models on the drug response

773 prediction task. PCC: Pearson correlation coefficient. SCC: Spearman
correlation coefficient. NES:

774 normalized enrichment score.

775

776

777

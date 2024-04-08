# ÚFAL LatinPipe at EvaLatin 2024: Morphosyntactic Analysis of Latin

This repository will contain the LatinPipe parser implementation described in
the _ÚFAL LatinPipe at EvaLatin 2024: Morphosyntactic Analysis of Latin_ paper.
The code will be published before the LT4HALA 2024 workshop, collocated with LREC-COLING 2024 in May 2024.

---

<img src="figures/LatinPipe.svg" alt="LatinPipe Architecture" align="right" style="width: 45%">

<h3 align="center">ÚFAL LatinPipe at EvaLatin 2024: Morphosyntactic Analysis of Latin</h3>

<p align="center">
  <b>Milan Straka</b> and <b>Jana Straková</b> and <b>Federica Gamba</b><br>
  Charles University<br>
  Faculty of Mathematics and Physics<br>
  Institute of Formal and Applied Lingustics<br>
  Malostranské nám. 25, Prague, Czech Republic
</p>

**Abstract:** We present LatinPipe, the winning submission to the EvaLatin 2024
Dependency Parsing shared task. Our system consists of a fine-tuned
concatenation of base and large pre-trained LMs, with a dot-product attention
head for parsing and softmax classification heads for morphology to jointly
learn both dependency parsing and morphological analysis. It is trained by
sampling from seven publicly available Latin corpora, utilizing additional
harmonization of annotations to achieve a more unified annotation style. Before
fine-tuning, we train the system for a few initial epochs with frozen weights.
We also add additional local relative contextualization by stacking the BiLSTM
layers on top of the Transformer(s). Finally, we ensemble output probability
distributions from seven randomly instantiated networks for the final
submission. The code is available at <a href="https://github.com/ufal/evalatin2024-latinpipe">https://github.com/ufal/evalatin2024-latinpipe</a>.<br clear="both">

---

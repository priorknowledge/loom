# Loom [![Build Status](https://travis-ci.org/priorknowledge/loom.svg?branch=master)](https://travis-ci.org/priorknowledge/loom)

Loom is a streaming inference and query engine for the
Cross-Categorization model \cite{mansinghka2009cross, shafto2011probabilistic}.

### Data Types

Loom learns models of <b>tabular data</b>, where hundreds of features are
partially observed over millions of rows.
Loom currently supports the following feature types and models:
- categoricals with up to 256 values as Dirichlet-Discrete
- unbounded categoricals as Dirichlet-Process-Discrete
- counts as Gamma-Poisson
- reals as Normal-Inverse-Chi-Squared-Normal

### Data Scale

Loom targets tabular datasets of sizes 100-1000 columns 10^3-10^9 rows.
To handle large datasets, loom implements <b>subsample annealing</b>
\cite{obermeyer2014scaling} with an accelerating annealing schedule and
adaptively turns off ineffective inference strategies.
Loom's annealing schedule is tuned to learn 10^6 row datasets in under an hour
and 10^9 row datasets in under a day.

<pre>
   Full Inference:     Partial Inference:  Greedy Inference:
   structure
   hyperparameters     hyperparameters
   mixtures            mixtures            mixtures
 |-------------------> ------------------> ------------------>
 1   many-passes   ~10^4   accelerate   10^9   single-pass  10^4
row                 rows                rows               row/sec
</pre>

## Documentation

* [Installation](/doc/installation.md)
* [Overview](/doc/overview.md)
* [Architecture](/doc/architecture.rst)
* [Inference Method](/doc/inference.rst)

## Authors

* Fritz Obermeyer <https://github.com/fritzo>
* Jonathan Glidden <https://twitter.com/jhglidden>

Loom is a streaming rewrite of the TARDIS engine developed by
Eric Jonas <https://twitter.com/stochastician> at Prior Knowledge, Inc.

Loom relies heavily on Salesforce.com's
[distributions](https://github.com/forcedotcom/distributions) library.

## References

<pre>
@inproceedings{mansinghka2009cross,
  title={Cross-categorization: A method for discovering multiple overlapping clusterings},
  author={Mansinghka, Vikash K and Jonas, Eric and Petschulat, Cap and Cronin, Beau and Shafto, Patrick and Tenenbaum, Joshua B},
  booktitle={Proc. of Nonparametric Bayes Workshop at NIPS},
  volume={2009},
  year={2009},
  url={http://web.mit.edu/vkm/www/crosscat.pdf},
}

@article{shafto2011probabilistic,
  title={A probabilistic model of cross-categorization},
  author={Shafto, Patrick and Kemp, Charles and Mansinghka, Vikash and Tenenbaum, Joshua B},
  journal={Cognition},
  volume={120},
  number={1},
  pages={1--25},
  year={2011},
  publisher={Elsevier}
  url={http://web.mit.edu/vkm/www/shaftokmt11_aprobabilisticmodelofcrosscategorization.pdf},
}

@ainproceedings{obermeyer2014scaling,
  title={Scaling Nonparametric Bayesian Inference via Subsample-Annealing},
  author={Obermeyer, Fritz and Glidden, Jonathan and Jonas, Eric},
  journal={JMLR Workshop and Conference Proceedings},
  editor={Samuel Kaski, Jukka Corander},
  year={2014},
  volume={33},
  url={http://arxiv.org/pdf/1402.5473v1.pdf},
}
</pre>

## License

Copyright (c) 2014 Salesforce.com, Inc. All rights reserved.

The PreQL query interface is covered by US patents pending:
* Application No. 14/014,204
* Application No. 14/014,221
* Application No. 14/014,225
* Application No. 14/014,236
* Application No. 14/014,241
* Application No. 14/014,250
* Application No. 14/014,258

Licensed under the Revised BSD License.
See [LICENSE.txt](LICENSE.txt) for details.

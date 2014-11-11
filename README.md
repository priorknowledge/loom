[![Build Status](https://travis-ci.org/priorknowledge/loom.svg?branch=master)](https://travis-ci.org/priorknowledge/loom)

# Loom

Loom is a streaming inference and query engine for the
Cross-Categorization model [mansinghka2009cross, shafto2011probabilistic](/doc/references.bib).

### Data Types

Loom learns models of <b>sparse heterogeneous tabular data</b>,
with hundreds of features and millions of rows.
Loom currently supports the following feature types and models:

* boolean fields as Beta-Bernoulli
* categorical fields with up to 256 values as Dirichlet-Discrete
* unbounded categorical fields as Dirichlet-Process-Discrete
* count fields as Gamma-Poisson
* real fields as Normal-Inverse-Chi-Squared-Normal
* sparse real fields as mixture of degenerate and dense real
* text and keyword fields as booleans for word absence/presence
* date fields as a combination of absolute, relative, and cyclic parts
* optional fields as a boolean plus one of the above feature models

See [input format docs](/doc/using.md#format) for details.

### Data Scale

Loom targets tabular datasets of sizes 100-1000 columns 10^3-10^9 rows.
To handle large datasets, loom implements <b>subsample annealing</b>
[obermeyer2014scaling](/doc/references.bib) with an accelerating annealing schedule and
adaptively turns off ineffective inference strategies.
Loom's annealing schedule is tuned to learn
10^8 cell datasets in under an hour and
10^10 cell datasets in under a day
(depending on feature type and sparsity).

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

* [Installing](/doc/installing.md)
* [Quick Start](/doc/quickstart.md)
* [Using Loom](/doc/using.md)
* [Adapting Loom](/doc/adapting.md)
* [Examples](/examples)

## Authors

* Fritz Obermeyer <https://github.com/fritzo>
* Jonathan Glidden <https://twitter.com/jhglidden>

Loom is a streaming rewrite of the TARDIS engine developed by
Eric Jonas <https://twitter.com/stochastician> at Prior Knowledge, Inc.

Loom relies heavily on Salesforce.com's
[distributions](https://github.com/forcedotcom/distributions) library.

## License

Copyright (c) 2014 Salesforce.com, Inc. All rights reserved.

Licensed under the Revised BSD License.
See [LICENSE.txt](LICENSE.txt) for details.

The PreQL query interface is covered by US patents pending:

* Application No. 14/014,204
* Application No. 14/014,221
* Application No. 14/014,225
* Application No. 14/014,236
* Application No. 14/014,241
* Application No. 14/014,250
* Application No. 14/014,258

### Dependencies

* [metis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) - Apache 2.0
* [pymetis](http://mathema.tician.de/software/pymetis) - MIT
* [numpy](http://docs.scipy.org/doc/numpy/license.html) - BSD
* [scipy](http://www.scipy.org/scipylib/license.html) - BSD
* [pandas](http://pandas.pydata.org) - BSD
* [matplotlib](http://matplotlib.org/users/license.html) - PSF
* [scikit-learn](http://scikit-learn.org) - BSD
* [simplejson](https://pypi.python.org/pypi/simplejson) - MIT
* [google protobuf](https://code.google.com/p/protobuf) - Apache 2.0
* [google perftools](https://code.google.com/p/gperftools) - New BSD
* [parsable](https://pypi.python.org/pypi/parsable) - MIT
* [distributions](https://github.com/forcedotcom/distributions) - Revised BSD
* [nose](https://pypi.python.org/pypi/nose) - LGPL
* [mock](https://pypi.python.org/pypi/mock) - New BSD

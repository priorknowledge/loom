# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from copy import copy
import csv
import math
from contextlib import contextmanager
from itertools import izip
from collections import Counter
from distributions.io.stream import json_load
from distributions.io.stream import open_compressed
from StringIO import StringIO
import numpy
from sklearn.cluster import SpectralClustering
from loom.format import load_decoder
from loom.format import load_encoder
import loom.store
import loom.query
import loom.group

SAMPLE_COUNT = 1000


class CsvWriter(object):
    def __init__(self, outfile, returns=None):
        writer = csv.writer(outfile)
        self.writerow = writer.writerow
        self.writerows = writer.writerows
        self.result = returns if returns else lambda: None


@contextmanager
def csv_output(arg):
    if arg is None:
        outfile = StringIO()
        yield CsvWriter(outfile, returns=outfile.getvalue)
    elif hasattr(arg, 'write'):
        yield CsvWriter(arg)
    else:
        with open_compressed(arg, 'w') as outfile:
            yield CsvWriter(outfile)


@contextmanager
def csv_input(arg):
    if hasattr(arg, 'read'):
        yield csv.reader(arg)
    else:
        with open_compressed(arg, 'rb') as infile:
            yield csv.reader(infile)


class PreQL(object):
    '''
    PreQL - Predictive Query Language server object.

    Data are assumed to be in csv format.  Data can be read from and written to
    file or can be passed around as StringIO objects.
    To convert among csv and pandas dataframes, use the transforms:

        input = StringIO(input_df.to_csv())  # input_df is a pandas.DataFrame
        output_df = pandas.read_csv(StringIO(output))

    Usage in scripts:

        with loom.preql.get_server('/absolute/path/to/dataset') as preql:
            preql.predict(...)
            preql.relate(...)
            preql.refine(...)
            preql.support(...)
            preql.group(...)

    Usage in iPython notebooks:

        preql = loom.preql.get_server('/absolute/path/to/dataset')

        preql.predict(...)
        preql.relate(...)
        preql.refine(...)
        preql.support(...)
        preql.group(...)

        preql.close()

    Methods:

        predict(rows_csv, count, result_out, ...)
            Draw samples from the posterior conditioned on rows in rows_csv.

        relate(columns, result_out, ...)
            Quantify dependency among columns and all other features.

        refine(target_feature_sets, query_feature_sets, conditioning_row, ...)
            Determine which queries would inform target features, in context.

        support(target_feature_sets, known_feature_sets, conditioning_row, ...)
            Determine which knolwedge has informed target features, in context.

        group(column, result_out)
            Cluster rows according to target column and related columns.

    Properties:

        feature_names - a list of all feature names
        converters - a dict of converters for use in pandas.read_csv
    '''
    def __init__(self, query_server, encoding=None, debug=False):
        self._paths = loom.store.get_paths(query_server.root)
        if encoding is None:
            encoding = self._paths['ingest']['encoding']
        self._query_server = query_server
        self._encoders = json_load(encoding)
        transforms = self._paths['ingest']['transforms']
        self._transform = loom.transforms.load_transforms(transforms)
        self._feature_names = [e['name'] for e in self._encoders]
        self._feature_set = frozenset(self._feature_names)
        self._name_to_pos = {
            name: i
            for i, name in enumerate(self._feature_names)
        }
        self._name_to_decode = {
            e['name']: load_decoder(e)
            for e in self._encoders
        }
        self._name_to_encode = {
            e['name']: load_encoder(e)
            for e in self._encoders
        }
        self._rowid_map = None
        self._debug = debug

    @property
    def feature_names(self):
        return self._feature_names[:]  # copy in lieu of frozenlist

    @property
    def converters(self):
        convert = lambda string: string if string else None
        return {name: convert for name in self._feature_names}

    @property
    def rowid_map(self):
        if self._rowid_map is None:
            filename = self._paths['ingest']['rowids']
            with loom.util.csv_reader(filename) as reader:
                self._rowid_map = {
                    int(internal_id): external_id
                    for internal_id, external_id in reader
                }
        return self._rowid_map

    def close(self):
        self._query_server.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def _cols_to_mask(self, cols):
        cols = set(cols)
        fnames = enumerate(self._feature_names)
        return frozenset(i for i, fname in fnames if fname in cols)

    def _validate_feature_set(self, feature_set):
        if len(feature_set) == 0:
            raise ValueError('empty feature set: '.format(feature_set))
        for name in feature_set:
            if name not in self._feature_set:
                raise ValueError('invalid feature: {}'.format(name))

    def _validate_feature_sets(self, feature_sets):
        for s in feature_sets:
            self._validate_feature_set(s)
        sets = set(feature_sets)
        if len(sets) != len(feature_sets):
            raise ValueError('duplicate sets in feature sets: {}'.format(sets))
        sum_len = sum(len(s) for s in feature_sets)
        len_sum = len(frozenset.union(*feature_sets))
        if sum_len != len_sum:
            raise ValueError('feature sets are not disjoint: {}'.format(sets))

    def _encode_row(self, row):
        if len(row) != len(self._feature_names):
            raise ValueError('invalid row (bad length): {}'.format(row))
        encoded_row = []
        for pos, value in enumerate(row):
            if value:
                assert isinstance(value, str), value
                encode = self._name_to_encode[self._feature_names[pos]]
                try:
                    encoded_row.append(encode(value))
                except:
                    raise ValueError(
                        'bad value at position {}: {}'.format(pos, value))
            else:
                encoded_row.append(None)
        return encoded_row

    def _decode_row(self, row):
        if len(row) != len(self._feature_names):
            raise ValueError('invalid row (bad length): {}'.format(row))
        decoded_row = []
        for pos, value in enumerate(row):
            if value is not None:
                decode = self._name_to_decode[self._feature_names[pos]]
                try:
                    decoded_row.append(decode(value))
                except:
                    raise ValueError(
                        'bad value at position {}: {}'.format(pos, value))
            else:
                decoded_row.append(None)
        return decoded_row

    def encode_set(self, feature_set):
        return self._feature_set & self._transform.forward_set(feature_set)

    def encode_row(self, row, header=None):
        features = self._feature_names
        if header is None:
            header = features
        if row is None:
            row = [None] * len(features)
        else:
            if isinstance(row, dict):
                row = self._transform.forward_dict(features, row)
            else:
                row = self._transform.forward_row(header, features, row)
            row = self._encode_row(row)
        return row

    def decode_row(self, row, header=None):
        features = self._feature_names
        if header is None:
            header = features
        row = self._decode_row(row)
        row = self._transform.backward_row(features, header, row)
        return row

    def _normalized_mutual_information(
            self,
            feature_set1,
            feature_set2,
            entropys=None,
            conditioning_row=None,
            sample_count=None):
        mi = self._query_server.mutual_information(
            feature_set1=feature_set1,
            feature_set2=feature_set2,
            entropys=entropys,
            conditioning_row=conditioning_row,
            sample_count=sample_count).mean
        return normalize_mutual_information(mi)

    def predict(self, rows_csv, count, result_out=None, id_offset=True):
        '''
        Samples from the conditional joint distribution.

        Inputs:
            rows_csv - filename/file handle/StringIO of input conditional rows
            count - number of samples to generate for each input row
            result_out - filename/file handle/StringIO of output samples,
                or None to return a csv string
            id_offset - whether to ignore column 0 as an unused id column

        Outputs:
            A csv with filled-in data rows sampled from the
            joint conditional posterior distribution.

        Example:
            Assume 'rows.csv' has already been written.

            >>> print open('rows.csv').read()
                feature0,feature1,feature2
                ,,
                0,,
                1,,
            >>> preql.predict('rows.csv', 2, 'result.csv', id_offset=False)
            >>> print open('result.csv').read()
                feature0,feature1,feature2
                0.5,0.1,True
                0.5,0.2,True
                0,1.5,False
                0,1.3,True
                1,0.1,False
                1,0.2,False
        '''
        with csv_output(result_out) as writer:
            with csv_input(rows_csv) as reader:
                self._predict(reader, count, writer, id_offset)
                return writer.result()

    def _predict(self, reader, count, writer, id_offset):
        header = reader.next()
        if id_offset and header[0] in self._feature_names:
            raise ValueError('id field conflict: {}'.format(header[0]))
        writer.writerow(header)
        for row in reader:
            if id_offset:
                row_id = row[0]
            conditioning_row = self.encode_row(row, header)
            to_sample = [value is None for value in conditioning_row]
            samples = self._query_server.sample(
                to_sample,
                conditioning_row,
                count)
            for sample in samples:
                print sample
                sample = self.decode_row(sample, header)
                if id_offset:
                    sample[0] = row_id
                writer.writerow(sample)

    def relate(self, columns, result_out=None, sample_count=SAMPLE_COUNT):
        '''
        Compute pairwise related scores between all pairs (f1,f2) of columns
        where f1 in input columns and f2 in all_features.

        Inputs:
            columns - a list of target feature names. a mix of fluent and basic
                features is allowed
            result_out - filename/file handle/StringIO of output relatedness,
                or None to return a csv string
            sample_count - number of samples in Monte Carlo computations;
                increasing sample_count increases accuracy

        Outputs:
            A csv with columns corresponding to input columns and one row
            per dataset feature.  The value in each cell is a relatedness
            number in [0,1] with 0 meaning independent and 1 meaning
            highly related.

            Related scores are defined in terms of mutual information via
            loom.preql.normalize_mutual_information.  For multivariate Gaussian
            data, relatedness equals Pearson's correlation; for non-Gaussian
            and discrete data, relatedness captures dependence in a more
            general way.

        Example:
            >>> print preql.relate(['feature0', 'feature2'])
            ,feature0,feature2
            feature0,1.0,0.5
            feature1,0.0,0.5
            feature2,0.5,1.0
        '''
        target_feature_sets = [self.encode_set([f]) for f in columns]
        query_feature_sets = [self.encode_set([f]) for f in columns]
        conditioning_row = self.encode_row(None)
        with csv_output(result_out) as writer:
            self._relate(
                target_feature_sets,
                query_feature_sets,
                conditioning_row,
                writer,
                sample_count)
            return writer.result()

    def refine(
            self,
            target_feature_sets=None,
            query_feature_sets=None,
            conditioning_row=None,
            result_out=None,
            sample_count=SAMPLE_COUNT):
        '''
        Determine which queries would inform target features, in context.

        Specifically, compute a matrix of values relatedness values

            [[r(t,q) for q in query_feature_sets] for t in target_feature_sets]

        conditioned on conditioning_row.

        Inputs:
            target_feature_sets - list of disjoint sets of feature names;
                defaults to [[f] for f unobserved in conditioning_row]
            query_feature_sets - list of disjoint sets of feature names;
                defaults to [[f] for f unobserved in conditioning_row]
            conditioning_row - a data row or dict of contextual information
            result_out - filename/file handle/StringIO of output data,
                or None to return a csv string
            sample_count - number of samples in Monte Carlo computations;
                increasing sample_count increases accuracy

        Outputs:
            A csv with columns corresponding to query_feature_sets and
            rows corresponding to target_feature_sets.  The value in each cell
            is a relatedness number in [0,1] with 0 meaning independent and 1
            meaning highly related.  See help(PreQL.relate) for details.
            Rows and columns will be labeled by the lexicographically-first
            feature in the respective set.

        Example:
            >>> print preql.refine(
                    [['f0', 'f1'], ['f2']],
                    [['f0', 'f1'], ['f2'], ['f3']],
                    [None, None, None, 1.0])
            ,f0,f2,f3
            f0,1.,0.9,0.5
            f2,0.8,1.,0.8
        '''
        conditioning_row = self.encode_row(conditioning_row)
        fc_zip = zip(self._feature_names, conditioning_row)
        if target_feature_sets is None:
            target_feature_sets = [[f] for f, c in fc_zip if c is None]
        if query_feature_sets is None:
            query_feature_sets = [[f] for f, c in fc_zip if c is None]
        target_feature_sets = map(self.encode_set, target_feature_sets)
        query_feature_sets = map(self.encode_set, query_feature_sets)
        unobserved_features = frozenset.union(*target_feature_sets) | \
            frozenset.union(*query_feature_sets)
        mismatches = []
        for feature, condition in fc_zip:
            if feature in unobserved_features and condition is not None:
                mismatches.append(feature)
        if mismatches:
            raise ValueError(
                'features {} must be None in conditioning row {}'.format(
                    mismatches,
                    conditioning_row))
        self._validate_feature_sets(target_feature_sets)
        self._validate_feature_sets(query_feature_sets)
        with csv_output(result_out) as writer:
            self._relate(
                target_feature_sets,
                query_feature_sets,
                conditioning_row,
                writer,
                sample_count)
            return writer.result()

    def support(
            self,
            target_feature_sets=None,
            observed_feature_sets=None,
            conditioning_row=None,
            result_out=None,
            sample_count=SAMPLE_COUNT):
        '''
        Determine which observed features most inform target features,
            in context.

        Specifically, compute a matrix of values relatedness values

            [[r(t,o | conditioning_row - o) for o in observed_feature_sets]
                for t in target_feature_sets]

        Where `conditioning_row - o` denotes the `conditioning_row`
            with feature `o` set to unobserved.

        Note that both features in observed and features in target
            must be observed in conditioning row.

        Inputs:
            target_feature_sets - list of disjoint sets of feature names;
                defaults to [[f] for f observed in conditioning_row]
            observed_feature_sets - list of disjoint sets of feature names;
                defaults to [[f] for f observed in conditioning_row]
            conditioning_row - a data row of contextual information
            result_out - filename/file handle/StringIO of output data,
                or None to return a csv string
            sample_count - number of samples in Monte Carlo computations;
                increasing sample_count increases accuracy

        Outputs:
            A csv with columns corresponding to observed_feature_sets and
            rows corresponding to target_feature_sets.  The value in each cell
            is a relatedness number in [0,1] with 0 meaning independent and 1
            meaning highly related.  See help(PreQL.relate) for details.
            Rows and columns will be labeled by the lexicographically-first
            feature in the respective set.

        Example:
            >>> print preql.support(
                    [['f0', 'f1'], ['f3']],
                    [['f0', 'f1'], ['f2'], ['f3']],
                    ['a', 7, None, 1.0])
            ,f0,f2,f3
            f0,1.,0.9,0.5
            f3,0.8,0.8,1.0
        '''
        conditioning_row = self.encode_row(conditioning_row)
        if all(c is None for c in conditioning_row):
            raise ValueError(
                'conditioning row must have at least one observation')
        fc_zip = zip(self._feature_names, conditioning_row)
        if target_feature_sets is None:
            target_feature_sets = [[f] for f, c in fc_zip if c is not None]
        if observed_feature_sets is None:
            observed_feature_sets = [[f] for f, c in fc_zip if c is not None]
        target_feature_sets = map(self.encode_set, target_feature_sets)
        observed_feature_sets = map(self.encode_set, observed_feature_sets)
        self._validate_feature_sets(target_feature_sets)
        self._validate_feature_sets(observed_feature_sets)
        observed_features = frozenset.union(*target_feature_sets) | \
            frozenset.union(*observed_feature_sets)
        mismatches = []
        for feature, condition in fc_zip:
            if feature in observed_features and condition is None:
                mismatches.append(feature)
        if mismatches:
            raise ValueError(
                'features {} must not be None in conditioning row {}'.format(
                    mismatches,
                    conditioning_row))
        with csv_output(result_out) as writer:
            self._relate(
                target_feature_sets,
                observed_feature_sets,
                conditioning_row,
                writer,
                sample_count)
            return writer.result()

    def _relate(
            self,
            target_feature_sets,
            query_feature_sets,
            conditioning_row,
            writer,
            sample_count):
        '''
        Compute all pairwise related scores between target_set
        and query_set

        In general it is assumed that all features in the target set
        and query set are unobserved in the conditioning row. If a feature
        is not unobserved, all related scores involving that feature will be
        computed with respect to a conditioning row with that feature set to
        unobserved.
        '''
        for tfs in target_feature_sets:
            for qfs in query_feature_sets:
                if tfs != qfs and tfs.intersection(qfs):
                    raise ValueError('target features and query features'
                                     ' must be disjoint or equal:'
                                     ' {} {}'.format(tfs, qfs))
        target_sets = map(self._cols_to_mask, target_feature_sets)
        query_sets = map(self._cols_to_mask, query_feature_sets)
        target_labels = map(min, target_feature_sets)
        query_labels = map(min, query_feature_sets)
        entropys = self._query_server.entropy(
            row_sets=target_sets,
            col_sets=query_sets,
            conditioning_row=conditioning_row,
            sample_count=sample_count)
        writer.writerow([None] + query_labels)
        for target_label, target_set in izip(target_labels, target_sets):
            result_row = [target_label]
            for query_set in query_sets:
                if target_set == query_set:
                    normalized_mi = 1.0
                else:
                    forgetful_conditioning_row = copy(conditioning_row)
                    for feature_index in target_set | query_set:
                        forgetful_conditioning_row[feature_index] = None
                    if forgetful_conditioning_row != conditioning_row:
                        normalized_mi = self._normalized_mutual_information(
                            target_set,
                            query_set,
                            entropys=None,
                            conditioning_row=forgetful_conditioning_row,
                            sample_count=sample_count)
                    else:
                        normalized_mi = self._normalized_mutual_information(
                            target_set,
                            query_set,
                            entropys=entropys,
                            sample_count=sample_count)
                result_row.append(normalized_mi)
            writer.writerow(result_row)

    def group(self, column, result_out=None):
        '''
        Compute consensus grouping for a single column.

        Inputs:
            column - name of a target feature to group by
            result_out - filename/file handle/StringIO of output groupings,
                or None to return a csv string

        Outputs:
            A csv file with columns [row_id, group_id, confidence]
            with one row per dataset row.  Confidence is a real number in [0,1]
            meaning how confident a row is to be in a given group.  Each row_id
            appears exactly once.  Csv rows are sorted lexicographically by
            group_id, then confidence.  Groupids are nonnegative integers.
            Larger groupids are listed first, so group 0 is the largest.

        Example:
            >>> print preql.group('feature0')
            row_id,group_id,confidence
            5,0,1.0
            3,0,0.5
            9,0,0.1
            2,1,0.9
            4,1,0.1
            0,2,0.4
        '''
        with csv_output(result_out) as writer:
            self._group(column, writer)
            return writer.result()

    def _group(self, column, writer):
        root = self._query_server.root
        feature_pos = self._name_to_pos[column]
        result = loom.group.group(root, feature_pos)
        rowid_map = self.rowid_map
        writer.writerow(loom.group.Row._fields)
        for row in result:
            external_id = rowid_map[row.row_id]
            writer.writerow((external_id, row.group_id, row.confidence))

    def similar(self, rows, rows2=None, row_limit=None, result_out=None):
        '''
        Compute pairwise similarity scores for all rows

        Inputs:
            rows - a list of data rows or dicts
            rows2 - a optional second list of data rows or dicts

        Outputs:
            A csv file with columns and rows denoting rows and
            entries ij giving the similarity score between row i
            and row j.
        '''
        rows = map(self.encode_row, rows)
        if rows2 is not None:
            rows2 = map(self.encode_row, rows2)
        else:
            rows2 = rows
        with csv_output(result_out) as writer:
            self._similar(rows, rows2, row_limit, writer)
            return writer.result()

    def _similar(self, update_rows, score_rows, row_limit, writer):
        score_ids = set()
        update_row_results = []
        for update_row in update_rows:
            results = self._query_server.score_derivative(
                update_row,
                score_rows,
                row_limit=row_limit)
            results_dict = dict(results)
            update_row_results.append(results_dict)
            score_ids = score_ids.union(set(results_dict.keys()))
        for results in update_row_results:
            writer.writerow([results[_id] for _id in score_ids])

    def search(self, row, row_limit=None, result_out=None):
        '''
        Find the top n most similar rows to `row` in the dataset.

        Inputs:
            row - a data row or dict

        Outputs
            A csv file with with columns row_id, score, showing the
            top 1000 most search rows in the dataset, sorted by score
        '''
        row = self.encode_row(row)
        with csv_output(result_out) as writer:
            self._search(row, row_limit, writer)
            return writer.result()

    def _search(self, row, row_limit, writer):
        results = self._query_server.score_derivative(
            row,
            score_rows=None,
            row_limit=row_limit)
        # FIXME map through erf
        writer.writerow(('row_id', 'score'))
        for row_id, score in results:
            external_id = self.rowid_map[row_id]
            writer.writerow((external_id, score))

    def cluster(
            self,
            rows_to_cluster=None,
            seed_rows=None,
            cluster_count=None,
            nearest_neighbors=10):
        if seed_rows is None:
            seed_rows = self._query_server.sample(
                [None for _ in self.feature_names],
                sample_count=SAMPLE_COUNT)
        row_limit = len(seed_rows) ** 2 + 1
        similar_string = StringIO(self.similar(seed_rows, row_limit=row_limit))
        similar = numpy.genfromtxt(
            similar_string,
            delimiter=',',
            skip_header=0)
        similar = similar.clip(0., 5.)
        similar = numpy.exp(similar)
        clustering = SpectralClustering(
            n_clusters=cluster_count,
            affinity='precomputed')
        labels = clustering.fit_predict(similar)

        if rows_to_cluster is None:
            return zip(labels, seed_rows)
        else:
            row_labels = []
            for row in rows_to_cluster:
                similar_scores = self.similar(
                    [row],
                    seed_rows,
                    row_limit=row_limit)
                similar_scores = numpy.genfromtxt(
                    StringIO(similar_scores),
                    delimiter=',',
                    skip_header=0)
                assert len(similar_scores) == len(labels)
                label_scores = zip(similar_scores, labels)
                top = sorted(label_scores, reverse=True)[:nearest_neighbors]
                label_counts = Counter(zip(*top)[1]).items()
                top_label = sorted(label_counts, key=lambda x: -x[1])[0][0]
                row_labels.append(top_label)
            return zip(row_labels, rows_to_cluster)


def normalize_mutual_information(mutual_info):
    '''
    Recall that mutual information

        I(X; Y) = H(X) + H(Y) - H(X, Y)

    satisfies:

        I(X; Y) >= 0
        I(X; Y) = 0 iff p(x, y) = p(x) p(y)  # independence

    Definition: Define the "relatedness" of X and Y by

        r(X, Y) = sqrt(1 - exp(-2 I(X; Y)))
                = sqrt(1 - exp(-I(X; Y))^2)
                = sqrt(1 - exp(H(X,Y) - H(X) - H(Y))^2)

    Theorem: Assume X, Y have finite entropy. Then
        (1) 0 <= r(X, Y) < 1
        (2) r(X, Y) = 0 iff p(x, y) = p(x) p(y)
        (3) r(X, Y) = r(Y, X)
    Proof: Abbreviate I = I(X; Y) and r = r(X, Y).
        (1) Since I >= 0, exp(-2 I) in (0, 1], and r in [0, 1).
        (2) r(X, Y) = 0  iff I(X; Y) = 0 iff p(x, y) = p(x) p(y)
        (3) r is symmetric since I is symmetric.                    []

    Theorem: If (X,Y) ~ MVN(mu, sigma_x, sigma_y, rho) in terms of
        standard deviations and Pearson's correlation coefficient,
        then r(X,Y) = rho.
    Proof: The covariance matrix is

            Sigma = [ sigma_x^2             sigma_x sigma_y rho ]
                    [ sigma_x sigma_y rho   sigma_y^2           ]

        with determinant det Sigma = sigma_x^2 sigma_y^2 (1 - rho^2).
        The mutual information is thus

            I(X;Y) = H(X) + H(Y) - H(X,Y)
                   = log(2 pi e)/2 + log sigma_x
                   + log(2 pi e)/2 + log sigma_y
                   - log(2 pi e) - 1/2 log det Sigma
                   = -1/2 log (1 - rho^2)
                   = -log sqrt(1 - rho^2)

        whence

            r(X,Y) = sqrt(1 - exp(-I(X;Y)) ** 2)
                   = sqrt(1 - exp(-2 I(X;Y)))
                   = rho                                            []
    '''
    mutual_info = max(mutual_info, 0)  # account for roundoff error
    r = (1.0 - math.exp(-2.0 * mutual_info)) ** 0.5
    assert 0 <= r and r < 1, r
    return r


def get_server(root, encoding=None, debug=False, profile=None, config=None):
    query_server = loom.query.get_server(root, config, debug, profile)
    return PreQL(query_server, encoding)

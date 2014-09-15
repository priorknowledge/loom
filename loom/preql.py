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

import csv
import math
from itertools import product
from distributions.io.stream import open_compressed, json_load
from cStringIO import StringIO
from loom.format import load_encoder, load_decoder
import loom.query
import loom.group

SAMPLE_COUNT = 1000


class PreQL(object):
    def __init__(self, query_server, encoding, debug=False):
        self._query_server = query_server
        self._encoders = json_load(encoding)
        self._feature_names = [e['name'] for e in self._encoders]
        self._name_to_pos = {
            name: i
            for i, name in enumerate(self._feature_names)
        }
        self._name_to_decode = {
            e['name']: load_decoder(e)
            for e in self._encoders
        }
        self._debug = debug

    def close(self):
        self._query_server.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    @property
    def feature_names(self):
        return self._feature_names[:]

    def predict(self, rows_csv, count, result_out, id_offset=True):
        '''
        Samples from the conditional joint distribution.

        Inputs:
            rows_csv - filename of input conditional rows csv
            count - number of samples to generate for each input row
            result_out - filename of output samples csv
            id_offset - whether to ignore column 0 as an unused id column

        Outputs:
            A csv file with filled-in data rows sampled from the
            joint conditional posterior distribution.

        Example:
            Assume 'rows.csv' has already been written.

            >>> print open('rows.csv').read()
                feature0,feature1,feature2
                ,,
                0,,
                1,,
            >>> preql.predict('rows.csv', 2, 'result.csv')
            >>> print open('result.csv').read()
                feature0,feature1,feature2
                0.5,0.1,True
                0.5,0.2,True
                0,1.5,False
                0,1.3,True
                1,0.1,False
                1,0.2,False
        '''
        with open_compressed(rows_csv, 'rb') as fin:
            with open_compressed(result_out, 'w') as fout:
                reader = csv.reader(fin)
                writer = csv.writer(fout)
                feature_names = list(reader.next())
                writer.writerow(feature_names)
                if id_offset:
                    feature_names.pop(0)
                name_to_pos = {name: i for i, name in enumerate(feature_names)}
                schema = []
                for encoder in self._encoders:
                    pos = name_to_pos.get(encoder['name'])
                    encode = load_encoder(encoder)
                    schema.append((pos, encode))
                for row in reader:
                    conditioning_row = []
                    to_sample = []
                    if id_offset:
                        row_id = row.pop(0)
                    for pos, encode, in schema:
                        value = None if pos is None else row[pos].strip()
                        observed = bool(value)
                        encoded = encode(value) if observed else None
                        conditioning_row.append(encoded)
                        to_sample.append(not observed)
                    samples = self._query_server.sample(
                        to_sample,
                        conditioning_row,
                        count)
                    for sample in samples:
                        out_row = [row_id] if id_offset else []
                        for name in feature_names:
                            pos = self._name_to_pos[name]
                            decode = self._name_to_decode[name]
                            val = sample[pos]
                            out_row.append(decode(val))
                        writer.writerow(out_row)

    def _cols_to_mask(self, cols):
        cols = set(cols)
        fnames = enumerate(self._feature_names)
        return frozenset(i for i, fname in fnames if fname in cols)

    def relate(self, columns, result_out=None, sample_count=SAMPLE_COUNT):
        '''
        Compute pairwise related scores between all pairs (f1,f2) of columns
        where f1 in input columns and f2 in all_features.

        Inputs:
            columns - a list of target feature names
            result_out - filename of output relatedness csv,
                or None to return a csv string
            sample_count - number of samples in Monte Carlo comutations;
                increasing sample_count increases accuracy

        Outputs:
            A csv file with columns corresponding to input columns and one row
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
        if result_out is None:
            outfile = StringIO()
            self._relate(columns, outfile, sample_count)
            return outfile.getvalue()
        else:
            with open_compressed(result_out, 'w') as outfile:
                self._relate(columns, outfile, sample_count)

    def _relate(self, columns, outfile, sample_count):
        fnames = self._feature_names
        writer = csv.writer(outfile)
        writer.writerow([None] + columns)
        joints = map(set, product(columns, fnames))
        singles = map(lambda x: {x}, columns + fnames)
        column_groups = singles + joints
        feature_sets = list(set(map(self._cols_to_mask, column_groups)))
        entropys = self._query_server.entropy(
            feature_sets,
            sample_count=sample_count)
        for to_relate in fnames:
            out_row = [to_relate]
            feature_sets1 = self._cols_to_mask({to_relate})
            for target_column in columns:
                if target_column == to_relate:
                    normalized_mi = 1.0
                else:
                    feature_sets2 = self._cols_to_mask({target_column})
                    mi = self._query_server.mutual_information(
                        feature_sets1,
                        feature_sets2,
                        entropys=entropys,
                        sample_count=sample_count).mean
                    normalized_mi = normalize_mutual_information(mi)
                out_row.append(normalized_mi)
            writer.writerow(out_row)

    def group(self, column, result_out=None):
        '''
        Compute consensus grouping for a single column.

        Inputs:
            column - name of a target feature to group by
            result_out - filename of output relatedness csv,
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
        if result_out is None:
            outfile = StringIO()
            self._group(column, outfile)
            return outfile.getvalue()
        else:
            with open_compressed(result_out, 'w') as outfile:
                self._group(column, outfile)

    def _group(self, column, output):
        root = self._query_server.root
        feature_pos = self._name_to_pos[column]
        result = loom.group.group(root, feature_pos)
        writer = csv.writer(output)
        writer.writerow(loom.group.Row._fields)
        for row in result:
            writer.writerow(row)


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


def get_server(root, encoding, debug=False, profile=None):
    query_server = loom.query.get_server(root, debug, profile)
    return PreQL(query_server, encoding)

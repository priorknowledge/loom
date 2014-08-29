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
from itertools import product
from distributions.io.stream import open_compressed, json_load
from cStringIO import StringIO
from loom.format import load_encoder, load_decoder
import loom.query
from loom.util import fixme


class PreQL(object):
    def __init__(self, query_server, encoding, debug=False):
        self.query_server = query_server
        self.encoders = json_load(encoding)
        self.feature_names = [e['name'] for e in self.encoders]
        self.name_to_pos = {name: i
                            for i, name in enumerate(self.feature_names)}
        self.name_to_decode = {e['name']: load_decoder(e)
                               for e in self.encoders}
        self.debug = debug

    def close(self):
        self.query_server.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def predict(self, rows_csv, count, result_out, id_offset=True):
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
                for encoder in self.encoders:
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
                        to_sample.append((not observed))
                        if observed is False:
                            conditioning_row.append(None)
                        else:
                            conditioning_row.append(encode(value))
                    samples = self.query_server.sample(
                        to_sample,
                        conditioning_row,
                        count)
                    samples = list(samples)
                    with open('samples.csv', 'w') as samplef:
                        csv.writer(samplef).writerows(samples)
                    for sample in samples:
                        if id_offset:
                            out_row = [row_id]
                        else:
                            out_row = []
                        for name in feature_names:
                            pos = self.name_to_pos[name]
                            decode = self.name_to_decode[name]
                            val = sample[pos]
                            out_row.append(decode(val))
                        writer.writerow(out_row)

    def cols_to_mask(self, cols):
        cols = set(cols)
        return tuple([fname in cols for fname in self.feature_names])

    def normalize_mutual_information(self, mutual_info, joint_entropy):
        """
        Mutual information is normalized by joint entopy because:
            I(X; Y) = 0 if p(x, y) = p(x)p(y) (independence)
            and
            I(X; Y) = H(X) + H(Y) - H(X, Y)
            H(X, X) = H(X)
            => I(X; X) = H(X) = H(X, X)

        FIXME(jglidden)
        This only works for discrete datatypes, since for real-valued data
        the entropy is arbitrarily shifted.  Specifically, if X ~ N(mu,signma),
        then H(X) can be any real number depending on sigma.
        """
        if not (joint_entropy > 0):
            raise fixme('jglidden', 'fix math error')
        return mutual_info / joint_entropy

    def relate(self, columns, result_out=None, sample_count=1000):
        """
        Compute pairwise related scores between all pairs of
        columns in columns.

        Related scores are defined to be:
            Related(X, Y) = I(X; Y) / H(X, Y)
        Where:
            I(X; Y) is the mutual information between X and Y:
                I(X; Y) = E[ log( p(x, y)) / ( p(x) p(y) ) ]; x, y ~ p(x, y)
            H(X) is the entropy of X:
                H(X) = E[ log( p(x) )]; x ~ p(x)
        Expectations are estimated via monte carlo with `sample_count` samples
        """
        if result_out is None:
            outfile = StringIO()
            self._relate(columns, outfile, sample_count)
            return outfile.getvalue()
        else:
            with open_compressed(result_out, 'w') as outfile:
                self._relate(columns, outfile, sample_count)

    def _relate(self, columns, outfile, sample_count):
        fnames = self.feature_names
        writer = csv.writer(outfile)
        writer.writerow(columns)
        joints = map(set, product(columns, fnames))
        singles = map(lambda x: {x}, columns + fnames)
        column_groups = singles + joints
        variable_masks = map(self.cols_to_mask, column_groups)
        entropys = self.query_server.entropy(
            variable_masks,
            sample_count=sample_count)
        for to_relate in fnames:
            out_row = [to_relate]
            variable_mask1 = self.cols_to_mask({to_relate})
            for target_column in columns:
                variable_mask2 = self.cols_to_mask({target_column})
                mi = self.query_server.mutual_information(
                    variable_mask1,
                    variable_mask2,
                    entropys=entropys,
                    sample_count=sample_count).mean
                joined = self.cols_to_mask({to_relate, target_column})
                joint_entropy = entropys[joined].mean
                normalized_mi = self.normalize_mutual_information(
                    mi,
                    joint_entropy)
                out_row.append(normalized_mi)
            writer.writerow(out_row)


def get_server(root, encoding, debug=False, profile=None):
    query_server = loom.query.get_server(root, debug, profile)
    return PreQL(query_server, encoding)

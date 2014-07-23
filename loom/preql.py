from loom.format import load_encoder, load_decoder
from distributions.io.stream import open_compressed, json_load
import csv


class PreQL(object):
    def __init__(self, query_server, encoding, debug=False):
        self.query_server = query_server
        self.encoders = json_load(encoding)
        self.feature_names = [e['name'] for e in self.encoders]
        self.debug = debug

    def predict(self, rows_csv, count, result_out, id_offset=True):
        with open_compressed(rows_csv, 'rb') as fin:
            with open_compressed(result_out, 'w') as fout:
                reader = csv.reader(fin)
                writer = csv.writer(fout)
                feature_names = list(reader.next())
                writer.writerow(feature_names)
                name_to_pos = {name: i for i, name in enumerate(feature_names)}
                pos_to_decode = {}
                schema = []
                for encoder in self.encoders:
                    pos = name_to_pos.get(encoder['name'])
                    encode = load_encoder(encoder)
                    decode = load_decoder(encoder)
                    if pos is not None:
                        pos_to_decode[pos] = decode
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
                    for sample in samples:
                        if id_offset:
                            out_row = [row_id]
                        else:
                            out_row = []
                        for name in feature_names:
                            pos = name_to_pos[name]
                            decode = pos_to_decode[pos]
                            val = sample[pos]
                            out_row.append(val)
                        writer.writerow(out_row)

    def cols_to_sample(self, cols):
        cols = set(cols)
        return [fname in cols for fname in self.feature_names]

    def normalize_mutual_information(self, mutual_info, joint_entropy):
        """
        Mutual information is normalized by joint entopy because:
            I(X; Y) = 0 if p(x, y) = p(x)p(y) (independence)
            and
            I(X; Y) = H(X) + H(Y) - H(X, Y)
            H(X, X) = H(X)
            => I(X; X) = H(X) = H(X, X)
        """
        return mutual_info / joint_entropy

    def relate(self, columns, result_out, sample_count=1000):
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
        with open_compressed(result_out, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.feature_names)
            for target_column in set(columns):
                out_row = [target_column]
                to_sample1 = self.cols_to_sample([target_column])
                for to_relate in self.feature_names:
                    to_sample2 = self.cols_to_sample([to_relate])
                    mi = self.query_server.mutual_information(
                        to_sample1,
                        to_sample2,
                        sample_count=sample_count).mean
                    joined = [to_relate, target_column]
                    to_sample_both = self.cols_to_sample(joined)
                    joint_entropy = self.query_server.entropy(
                        to_sample_both,
                        sample_count=sample_count).mean
                    normalized_mi = self.normalize_mutual_information(
                        mi,
                        joint_entropy)
                    out_row.append(normalized_mi)
                writer.writerow(out_row)

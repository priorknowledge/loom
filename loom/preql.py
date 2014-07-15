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
            with open(result_out, 'w') as fout:
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
        return [fname in cols for fname in self.feature_names]

    def relate(self, columns, result_out, sample_count=1000):
        with open(result_out, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.feature_names)
            for target_column in columns:
                out_row = [target_column]
                to_sample1 = self.cols_to_sample([target_column])
                for to_relate in self.feature_names:
                    to_sample2 = self.cols_to_sample([to_relate])
                    mi = self.query_server.mutual_information(
                        to_sample1,
                        to_sample2,
                        sample_count=sample_count)
                    joined = [to_relate, target_column]
                    to_sample_both = self.cols_to_sample(joined)
                    ent = self.query_server.entropy(
                        to_sample_both,
                        sample_count=sample_count)
                    out_row.append(mi / (2 * ent))
                writer.writerow(out_row)

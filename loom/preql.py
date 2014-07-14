from loom.format import load_encoder, load_decoder
from distributions.io.stream import open_compressed, json_load
import csv


class PreQL(object):
    def __init__(self, query_server, encoding, debug=False):
        self.query_server = query_server
        self.encoders = json_load(encoding)
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
                for i, row in enumerate(reader):
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
                    for c, sample in enumerate(samples):
                        if id_offset:
                            out_row = ['{}_{}_{}'.format(row_id, i, c)]
                        else:
                            out_row = []
                        for name in feature_names:
                            pos = name_to_pos[name]
                            decode = pos_to_decode[pos]
                            val = sample[pos]
                            out_row.append(val)
                        writer.writerow(out_row)

'''
Emulate the tardis runner.
'''

import os
from distributions.io.stream import json_load, json_dump
from distributions.fileutil import tempdir
import loom.format
import loom.config
import loom.runner


class Runner(object):
    def __init__(
            self,
            config,
            meta,
            databin,
            datamask,
            latent,
            log_config=None,
            start_iter=0):
        self.config = config
        self.meta = meta
        self.data = databin
        self.mask = datamask
        self.latent = latent
        self.log_config = log_config
        self.start_iter = start_iter

    def run(self, ITERS):
        raise NotImplementedError('TODO')

    def run_default(self):
        iters = self.config['iters'] - self.progress.iters
        if iters > 0:
            self.run(iters)

    def write_latent(self, outfile, with_suffstats=False):
        raise NotImplementedError('TODO')

    @property
    def progress(self):
        return {
            'iters': None,
            'compute_cost_completed': None,
            'compute_cost_remaining': None,
        }

    def dump_state(self, sample_out, scores_out):
        raise NotImplementedError('TODO')
        json_dump({}, scores_out)


def run(config,
        meta,
        data,
        mask,
        latent,
        sampleout,
        scoreout,
        log_config=None):
    #runner = Runner(config, meta, databin, datamask, latent, log_config)
    #runner.run_default()
    #runner.dump_state(sampleout, scoreout)

    if log_config is None:
        log_config = {}
    elif isinstance(log_config, str):
        log_config = json_load(log_config)
    tags = log_config.get('tags', {})
    log_out = log_config.get('log_file', None)

    config = os.path.abspath(config)
    meta = os.path.abspath(meta)
    data = os.path.abspath(data)
    mask = os.path.abspath(mask)
    latent = os.path.abspath(latent)
    sampleout = os.path.abspath(sampleout)
    scoreout = os.path.abspath(scoreout)

    with tempdir():
        rows = os.path.abspath('rows.pbs.gz')
        config_in = os.path.abspath('config.pb.gz')
        model_in = os.path.abspath('model_in.pb.gz')
        model_out = os.path.abspath('model_out.pb.gz')
        groups_out = os.path.abspath('groups_out')
        assign_out = os.path.abspath('assign_out.pbs.gz')
        log = os.path.abspath('log.pbs.gz')

        print 'importing config'
        conf = json_load(config)
        loom.config.config_dump(conf, config_in)

        print 'importing latent'
        loom.format.import_latent(
            meta_in=meta,
            latent_in=latent,
            tardis_conf_in=config,
            model_out=model_in)

        print 'importing data'
        loom.format.import_data(
            meta_in=meta,
            data_in=data,
            mask_in=mask,
            rows_out=rows)

        print 'shuffling data with seed {}'.format(conf['seed'])
        loom.runner.shuffle(
            rows_in=rows,
            rows_out=rows,
            seed=conf['seed'])

        print 'inferring latent'
        loom.runner.infer(
            config_in=config_in,
            rows_in=rows,
            model_in=model_in,
            model_out=model_out,
            groups_out=groups_out,
            assign_out=assign_out,
            log_out=log)

        print 'exporting latent'
        loom.format.export_latent(
            meta_in=meta,
            model_in=model_out,
            groups_in=groups_out,
            assign_in=assign_out,
            latent_out=sampleout)

        print 'exporting scores'
        json_dump({}, scoreout)

        print 'exporting log'
        loom.format.export_log(log_in=log, log_out=log_out, **tags)

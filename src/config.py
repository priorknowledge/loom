from copy import deepcopy
import simplejson as json
import parsable
from distributions.io.stream import open_compressed
import loom.schema_pb2
parsable = parsable.Parsable()

DEFAULTS = {
    'seed': 0,
    'schedule': {
        'extra_passes': 500.0,
        'small_data_size': 4e3,
        'big_data_size': 1e9,
        'max_reject_iters': 100,
        'checkpoint_period_sec': 1e9,
    },
    'kernels': {
        'cat': {
            'empty_group_count': 1,
            'row_queue_capacity': 200,
            'parser_threads': 6,
        },
        'hyper': {
            'run': True,
            'parallel': True,
        },
        'kind': {
            'iterations': 32,
            'empty_kind_count': 32,
            'row_queue_capacity': 200,
            'parser_threads': 6,
            'score_parallel': True,
        },
    },
    'posterior_enum': {
        'sample_count': 100,
        'sample_skip': 10,
    },
    'generate': {
        'row_count': 0,
        'density': 0.0,
    }
}


def fill_in_defaults(config, defaults=DEFAULTS):
    assert isinstance(config, dict), config
    assert isinstance(defaults, dict), defaults
    for key, default in defaults.iteritems():
        if key not in config:
            config[key] = deepcopy(default)
        elif isinstance(default, dict):
            fill_in_defaults(config[key], default)


def protobuf_dump(config, message, warn='WARN ignoring config'):
    for key, value in config.iteritems():
        warn_key = '{}.{}'.format(warn, key) if warn else None
        if hasattr(message, key):
            if isinstance(value, dict):
                protobuf_dump(value, getattr(message, key), warn_key)
            else:
                setattr(message, key, value)
        elif warn:
            print warn_key


def config_dump(config, filename):
    config = deepcopy(config)
    fill_in_defaults(config)
    message = loom.schema_pb2.Config()
    protobuf_dump(config, message)
    with open_compressed(filename, 'wb') as f:
        f.write(message.SerializeToString())


if __name__ == '__main__':
    print json.dumps(DEFAULTS, indent=4, sort_keys=True)

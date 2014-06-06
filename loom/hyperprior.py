import numpy
import loom.gridding

# FIXME expand and test logspace prior
#dd_alpha = numpy.logspace(-1, 1, 10).tolist()  # TODO test on datasets
dd_alpha = numpy.linspace(0.1, 10, 10).tolist()  # DEPRECATED

pos_logspace = numpy.logspace(-8, 8, 100).tolist()
neg_logspace = (-numpy.logspace(-8, 8, 100)).tolist()

clustering = loom.gridding.pitman_yor()

DEFAULTS = {
    'outer_prior': clustering,
    'inner_prior': {
        'clustering': clustering,
        'bb': {
            'alpha': dd_alpha,
            'beta': dd_alpha,
        },
        'dd': {
            'alpha': dd_alpha,
        },
        'dpd': {
            'gamma': (10 ** loom.gridding.left_heavy(-1, 2, 30)).tolist(),
            'alpha': (10 ** loom.gridding.right_heavy(-1, 1, 20)).tolist(),
        },
        'gp': {
            'alpha': numpy.logspace(-1, 5, 100).tolist(),
            'inv_beta': numpy.logspace(-5, 1, 100).tolist(),
        },
        'nich': {
            'mu': neg_logspace + [0] + pos_logspace,
            'sigmasq': pos_logspace,
            'kappa': numpy.logspace(-2, 2, 30).tolist(),
            'nu': numpy.logspace(0, 2, 30).tolist(),
        },
    },
}


def dump_default(message):
    loom.util.dict_to_protobuf(DEFAULTS, message)

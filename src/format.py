#import loom.schema_pb2
#import distributions.schema_pb2
#import distributions.fileutil
#import simplejson as json
import parsable
parsable = parsable.Parsable()


@parsable.command
def import_latent(
        meta_in,
        latent_in,
        model_out,
        groups_out=None,
        assignments_out=None):
    '''
    Import latent from tardis format.
    '''
    raise NotImplementedError('TODO')


@parsable.command
def export_latent(
        meta_in,
        model_in,
        groups_in,
        latent_out,
        assignments_in=None):
    '''
    Export latent to tardis format.
    '''
    raise NotImplementedError('TODO')


@parsable.command
def import_data(
        meta_in,
        data_in,
        mask_in,
        values_out):
    '''
    Import dataset from tardis format.
    '''
    raise NotImplementedError('TODO')


@parsable.command
def export_data(
        meta_in,
        values_in,
        rows_out):
    '''
    Export dataset to tarot format.
    '''
    raise NotImplementedError('TODO')


if __name__ == '__main__':
    parsable.dispatch()

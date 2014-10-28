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

import os
import csv
import shutil
import random
from collections import Counter, defaultdict
from StringIO import StringIO
import numpy
import numpy.random
import scipy
import scipy.misc
import scipy.ndimage
import pandas
from matplotlib import pyplot
from sklearn.cluster import SpectralClustering
from distributions.dbg.random import sample_discrete
from distributions.io.stream import open_compressed
import loom.tasks
import loom.query
import loom.preql
import loom.store
import loom.datasets
import parsable
parsable = parsable.Parsable()


NAME = 'fox'
ROOT = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(ROOT, 'schema.json')
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results')
SAMPLES = os.path.join(DATA, 'samples.csv.gz')
IMAGE = scipy.misc.imread(os.path.join(ROOT, 'fox.png'))
ROW_COUNT = 10000
PASSES = 10
EMPTY_GROUP_COUNT = 10

SIMILAR = os.path.join(DATA, 'cluster_labels.csv.gz')

X_SCALE = 2.0 / (IMAGE.shape[0] - 1)
Y_SCALE = 2.0 / (IMAGE.shape[1] - 1)

for dirname in [DATA, RESULTS]:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def to_image_coordinates(loom_x, loom_y):
    x = int(round((loom_x + 1.0) / X_SCALE))
    y = int(round((loom_y + 1.0) / Y_SCALE))
    return x, y


def to_loom_coordinates(image_x, image_y):
    x = image_x * X_SCALE - 1.0
    y = image_y * Y_SCALE - 1.0
    return x, y


def sample_from_image(image, row_count):
    image = -1.0 * image
    image -= image.min()
    x_pmf = image.sum(axis=1)
    y_pmfs = image.copy()
    for y_pmf in y_pmfs:
        y_pmf /= (y_pmf.sum() + 1e-8)

    for _ in xrange(row_count):
        x = sample_discrete(x_pmf)
        y = sample_discrete(y_pmfs[x])
        x += numpy.random.random() - 0.5
        y += numpy.random.random() - 0.5
        yield to_loom_coordinates(x, y)


def synthesize_search(name, image_pos):
    shape = IMAGE.shape
    image = IMAGE.reshape(shape[0], shape[1], 1).repeat(3, 2)
    image[image_pos] = [0, 255, 0]
    with open_compressed(SAMPLES) as f:
        rows = list(csv.reader(f))[1:]
        rows = [map(float, r) for r in rows]
    root = loom.store.get_paths(name)['root']
    with loom.preql.get_server(root) as server:
        x, y = to_loom_coordinates(*image_pos)
        search = server.search((str(x), str(y)))
    search = csv.reader(StringIO(search))
    search.next()
    for row_id, score in search:
        score = numpy.exp(float(score))
        if score < 1.:
            return image
        row_id = int(row_id.split(':')[1])
        sample_x, sample_y = rows[row_id]
        x, y = to_image_coordinates(sample_x, sample_y)
        image[x, y] = [255 * (1 - 1/score), 0, 0]
    return image


def create_similar_matrix(name, sample_count):
    with open_compressed(SAMPLES) as f:
        reader = csv.reader(f)
        reader.next()
        samples = map(tuple, reader)
        pts = random.sample(samples, sample_count)

    row_limit = sample_count ** 2 + 1
    root = loom.store.get_paths(name)['root']
    print 'computing similar for {} samples'.format(sample_count)
    with loom.preql.get_server(root) as server:
        similar_scores = server.similar(
                pts,
                row_limit=row_limit)
    similar_df = pandas.DataFrame.from_csv(StringIO(similar_scores))
    df = pandas.DataFrame(pts, columns=['x', 'y'])
    similar_df.index = df.index
    df = df.join(similar_df)
    return df


def synthesize_clusters(name, similar_df, cluster_count, pixel_count):
    with open_compressed(SAMPLES) as f:
        reader = csv.reader(f)
        reader.next()
        samples = map(tuple, reader)
        total_sample_count = len(samples)
        samples = random.sample(samples, pixel_count)

    print 'finding clusters'
    pts = list(similar_df.to_records()[['x', 'y']])

    similar_df = similar_df[similar_df.columns[2:]]
    # HACK: fix this in query server
    similar_df *= total_sample_count / len(similar_df)
    similar_df = similar_df.clip_upper(5.)
    similar_df = numpy.exp(similar_df)

    sc = SpectralClustering(n_clusters=cluster_count, affinity='precomputed')
    labels = sc.fit_predict(numpy.array(similar_df))
    label_count = len(set(labels))

    print 'assigning {} samples to clusters'.format(len(samples))
    root = loom.store.get_paths(name)['root']
    row_limit = pixel_count * len(pts) + 1
    sample_labels = []
    with loom.preql.get_server(root) as server:
        for sample in samples:
            similar_scores = server.similar([sample], pts, row_limit=row_limit)
            similar_scores = pandas.DataFrame.from_csv(StringIO(similar_scores))
            top = sorted(zip(similar_scores.iloc[0], labels), reverse=True)[:10]
            top_label = sorted(Counter(zip(*top)[1]).items(), key=lambda x: -x[1])[0][0]
            sample_labels.append(top_label)

    shape = IMAGE.shape
    image = IMAGE.reshape(shape[0], shape[1], 1).repeat(3, 2)
    colors = pyplot.cm.Set1(numpy.linspace(0, 1, label_count))
    colors = (255 * colors[:, :3]).astype(numpy.uint8)
    print Counter(labels)
    print Counter(sample_labels)
    for sample, label in zip(samples, sample_labels):
        x, y = to_image_coordinates(float(sample[0]), float(sample[1]))
        image[x, y] = colors[label]
    print 'done'
    return image


def synthesize_image(name):
    print 'synthesizing image'
    width, height = IMAGE.shape
    image = numpy.zeros((width, height))
    root = loom.store.get_paths(name)['root']
    with loom.query.get_server(root) as server:
        for x in xrange(width):
            for y in xrange(height):
                xy = to_loom_coordinates(x, y)
                image[x, y] = server.score(xy)

    numpy.exp(image, out=image)
    image /= image.max()
    image -= 1.0
    image *= -255
    return image.astype(numpy.uint8)


def visualize_dataset(samples):
    width, height = IMAGE.shape
    image = numpy.zeros((width, height))
    for x, y in samples:
        x, y = to_image_coordinates(x, y)
        image[x, y] += 1
    image = scipy.ndimage.gaussian_filter(image, sigma=1)
    image *= -255.0 / image.max()
    image -= image.min()
    return image.astype(numpy.uint8)


@parsable.command
def create_dataset(row_count=ROW_COUNT):
    '''
    Extract dataset from image.
    '''
    scipy.misc.imsave(os.path.join(RESULTS, 'original.png'), IMAGE)
    print 'sampling {} points from image'.format(row_count)
    with open_compressed(SAMPLES, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for row in sample_from_image(IMAGE, row_count):
            writer.writerow(row)
    with open_compressed(SAMPLES) as f:
        reader = csv.reader(f)
        reader.next()
        image = visualize_dataset(map(float, row) for row in reader)
    scipy.misc.imsave(os.path.join(RESULTS, 'samples.png'), image)


@parsable.command
def compress(sample_count=1):
    '''
    Compress image using loom.
    '''
    assert os.path.exists(SAMPLES), 'first create dataset'
    print 'inferring'
    loom.tasks.ingest(NAME, SCHEMA, SAMPLES)
    loom.tasks.infer(NAME, sample_count=sample_count)
    image = synthesize_image(NAME)
    scipy.misc.imsave(os.path.join(RESULTS, 'loom.png'), image)


@parsable.command
def search(x=50, y=50):
    '''
    Demonstrate loom's search command.
    Highlight points search to the point (x, y)
    '''
    assert loom.store.get_paths(NAME)['samples'], 'first compress image'
    x = int(x)
    y = int(y)
    print 'finding points similar to {} {}'.format(x, y)
    image = synthesize_search(NAME, (x, y))
    scipy.misc.imsave(os.path.join(RESULTS, 'search.png'), image)


@parsable.command
def similar(sample_count):
    '''
    Create a similarity matrix from sample_count samples from the dataset
    '''
    sample_count = int(sample_count)
    clustering = create_similar_matrix(NAME, sample_count)
    with open_compressed(SIMILAR, 'w') as f:
        clustering.to_csv(f)


@parsable.command
def plot_clusters(cluster_count, pixel_count):
    '''
    Plot clusters
    '''
    cluster_count = int(cluster_count)
    pixel_count = int(pixel_count)
    with open_compressed(SIMILAR, 'r') as f:
        similar = pandas.DataFrame.from_csv(StringIO(f.read()))
    image = synthesize_clusters(NAME, similar, cluster_count, pixel_count)
    scipy.misc.imsave(os.path.join(RESULTS, 'cluster.png'), image)


@parsable.command
def cluster(cluster_count=5, sample_count=1000, pixel_count=None):
    '''
    Draw a fox map
    '''
    cluster_count = int(cluster_count)
    sample_count = int(sample_count)
    if pixel_count is None:
        with open_compressed(SAMPLES) as f:
            reader = csv.reader(f)
            pixel_count = len(list(reader)) - 1
    else:
        pixel_count = int(pixel_count)
    assert loom.store.get_paths(NAME)['samples'], 'first compress image'

    similar = create_similar_matrix(NAME, sample_count)
    image = synthesize_clusters(name, similar, cluster_count, pixel_count)
    scipy.misc.imsave(os.path.join(RESULTS, 'cluster.png'), image)


@parsable.command
def clean():
    '''
    Clean out dataset and results.
    '''
    for dirname in [DATA, RESULTS]:
        if not os.path.exists(dirname):
            shutil.rmtree(dirname)
    loom.datasets.clean(NAME)


@parsable.command
def run(row_count=ROW_COUNT, sample_count=1):
    '''
    Generate all datasets and run all algorithms.
    See index.html for results.
    '''
    create_dataset(row_count)
    compress(sample_count)
    print 'see file://{} for results'.format(os.path.join(ROOT, 'index.html'))


if __name__ == '__main__':
    parsable.dispatch()

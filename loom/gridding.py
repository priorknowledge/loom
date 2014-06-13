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

import numpy


def uniform(min_val, max_val, point_count):
    grid = numpy.array(range(point_count)) + 0.5
    grid *= (max_val - min_val) / float(point_count)
    grid += min_val
    return grid


def center_heavy(min_val, max_val, point_count):
    grid = uniform(-1, 1, point_count)
    grid = numpy.arcsin(grid) / numpy.pi + 0.5
    grid *= max_val - min_val
    grid += min_val
    return grid


def left_heavy(min_val, max_val, point_count):
    grid = uniform(0, 1, point_count)
    grid = grid ** 2
    grid *= max_val - min_val
    grid += min_val
    return grid


def right_heavy(min_val, max_val, point_count):
    grid = left_heavy(max_val, min_val, point_count)
    return grid[::-1].copy()


def pitman_yor(
        min_alpha=0.1,
        max_alpha=100,
        min_d=0,
        max_d=0.5,
        alpha_count=20,
        d_count=10):
    '''
    For d = 0, this degenerates to the CRP, where the expected number of
    tables is:
        E[table_count] = O(alpha log(customer_count))
    '''
    min_alpha = float(min_alpha)
    max_alpha = float(max_alpha)
    min_d = float(min_d)
    max_d = float(max_d)

    lower_triangle = [
        (x, y)
        for x in center_heavy(0, 1, alpha_count)
        for y in left_heavy(0, 1, d_count)
        if x + y < 1
    ]
    alpha = lambda x: min_alpha * (max_alpha / min_alpha) ** x
    d = lambda y: min_d + (max_d - min_d) * y
    grid = [
        {'alpha': alpha(x), 'd': d(y)}
        for (x, y) in lower_triangle
    ]
    return grid

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import faiss
from faiss.contrib.ondisk import merge_ondisk

#################################################################
# Small I/O functions
#################################################################


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


#################################################################
# Main program
#################################################################

stage = int(sys.argv[1])

tmpdir = '/tmp/'

if stage == 0:
    # train the index
    xt = fvecs_read("sift1M/sift_learn.fvecs")
    index = faiss.index_factory(xt.shape[1], "IVF4096,Flat")
    print("training index")
    index.train(xt)
    print("write " + tmpdir + "trained.index")
    faiss.write_index(index, tmpdir + "trained.index")


if 1 <= stage <= 4:
    # add 1/4 of the database to 4 independent indexes
    bno = stage - 1
    xb = fvecs_read("sift1M/sift_base.fvecs")
    i0, i1 = int(bno * xb.shape[0] / 4), int((bno + 1) * xb.shape[0] / 4)
    index = faiss.read_index(tmpdir + "trained.index")
    print("adding vectors %d:%d" % (i0, i1))
    index.add_with_ids(xb[i0:i1], np.arange(i0, i1))
    print("write " + tmpdir + "block_%d.index" % bno)
    faiss.write_index(index, tmpdir + "block_%d.index" % bno)

if stage == 5:

    print('loading trained index')
    # construct the output index
    index = faiss.read_index(tmpdir + "trained.index")

    block_fnames = [
        tmpdir + "block_%d.index" % bno
        for bno in range(4)
    ]

    merge_ondisk(index, block_fnames, tmpdir + "merged_index.ivfdata")

    print("write " + tmpdir + "populated.index")
    faiss.write_index(index, tmpdir + "populated.index")


if stage == 6:
    # perform a search from disk
    print("read " + tmpdir + "populated.index")
    # For OnDiskInvertedLists, we should use IO_FLAG_ONDISK_SAME_DIR instead of IO_FLAG_MMAP
    # because IO_FLAG_MMAP includes IO_FLAG_SKIP_IVF_DATA which prevents proper mapping
    try:
        IO_FLAG_ONDISK_SAME_DIR = faiss.IO_FLAG_ONDISK_SAME_DIR
        IO_FLAG_READ_ONLY = faiss.IO_FLAG_READ_ONLY
        io_flags = IO_FLAG_ONDISK_SAME_DIR | IO_FLAG_READ_ONLY
    except AttributeError:
        # Fallback: try IO_FLAG_ONDISK_SAME_DIR alone, or use no flags
        try:
            io_flags = faiss.IO_FLAG_ONDISK_SAME_DIR
        except AttributeError:
            # If flags are not available, read without flags (should still work)
            io_flags = 0
    
    index = faiss.read_index(tmpdir + "populated.index", io_flags)
    index.nprobe = 16

    # load query vectors and ground-truth
    xq = fvecs_read("sift1M/sift_query.fvecs")
    gt = ivecs_read("sift1M/sift_groundtruth.ivecs")

    D, I = index.search(xq, 5)

    recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(xq.shape[0])
    print("recall@1: %.3f" % recall_at_1)
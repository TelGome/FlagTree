# Copyright 2026-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Minimal TLE distributed kernel used by the lowering demo.

On Ascend, ``triton.experimental.tle.language`` swaps the distributed entry
points for ``tle/language/dsa/ascend/distributed.py``, which emits the same
``distributed.*`` ops as ``triton_dist.language`` (dl.*):

    tle.shard_id            -> distributed.get_rank   (dl.rank)
    tle.remote              -> distributed.symm_at    (dl.symm_at)
    tle.distributed_barrier -> distributed.extern_call(symbol="aclshmem_barrier_all")
"""

import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def ring_exchange(x_ptr, out_ptr):
    v = tl.load(x_ptr)
    rank = tle.shard_id(axis=-1)
    tle.distributed_barrier()
    peer = (rank + 1) % 8
    remote_ptr = tle.remote(x_ptr, peer)
    tl.store(remote_ptr, v)
    tl.store(out_ptr, v)

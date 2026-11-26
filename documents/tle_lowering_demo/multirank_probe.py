# Copyright 2026-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Multi-rank runtime probe for the Ascend TLE distributed primitives.

Run (needs exclusive access to the cards):

    ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=8990 \
      torchrun --nproc-per-node=4 multirank_probe.py

Checks, per rank:
  1. tle.shard_id() == torch.distributed rank
  2. tle.n_pes()    == world size
  3. tle.distributed_barrier() orders every shard write before the remote read
  4. tle.remote(peer) + load reads the peer's shard (device space)
  5. tle.signal(op="inc") from every rank counters up to world
  6. tle.signal(op="add", value=rank+1) from every rank counters up to the sum
"""

import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle as tle

N = 8
SIG_ELEMS = 1024  # >= slot_id * 16 (64B per slot / 4B per i32)


@triton.jit
def probe(data_ptr, sig_ptr, out_ptr, rank_out, npes_out, NPES: tl.constexpr, N: tl.constexpr):
    offs = tl.arange(0, N)

    rank = tle.shard_id(axis=-1)
    npes = tle.n_pes()

    # Make every rank's local shard visible before anyone reads it remotely.
    tle.distributed_barrier()

    peer = (rank + 1) % npes
    remote = tle.remote(data_ptr, space="device", dtype=tl.float32, shard_id=peer)
    tl.store(out_ptr + offs, tl.load(remote + offs))

    tl.store(rank_out, rank)
    tl.store(npes_out, npes)

    # slot 0: every rank increments every peer's counter by 1 -> NPES
    # slot 1: every rank adds rank+1 to every peer's counter   -> NPES*(NPES+1)/2
    for p in tl.static_range(0, NPES):
        tle.signal(sig_ptr, p, slot_id=0, op="inc")
        tle.signal(sig_ptr, p, slot_id=1, op="add", value=rank + 1)

    tle.signal_wait(sig_ptr, slot_id=0, wait_kind="signal", target=NPES)
    tle.signal_wait(sig_ptr, slot_id=1, wait_kind="signal", target=(NPES * (NPES + 1)) // 2)


def main():
    tle.init_communicator()
    rank = dist.get_rank()
    world = dist.get_world_size()

    x = torch.arange(N, dtype=torch.float32, device="npu") + rank * 1000
    data = tle.create_dist_tensor(x)
    data.copy_(x)

    sig = tle.create_dist_tensor(torch.zeros(SIG_ELEMS, dtype=torch.int32, device="npu"))
    sig.zero_()

    out = torch.full((N, ), -1.0, dtype=torch.float32, device="npu")
    rank_out = torch.full((1, ), -1, dtype=torch.int32, device="npu")
    npes_out = torch.full((1, ), -1, dtype=torch.int32, device="npu")

    probe[(1,)](data, sig, out, rank_out, npes_out, NPES=world, N=N)
    torch.npu.synchronize()

    peer = (rank + 1) % world
    expected = torch.arange(N, dtype=torch.float32, device="npu") + peer * 1000
    host = out.cpu()
    got_rank = int(rank_out.cpu().item())
    got_npes = int(npes_out.cpu().item())
    slot0 = int(sig[0].cpu().item())
    slot1 = int(sig[16].cpu().item())

    checks = [
        ("shard_id", got_rank, rank),
        ("n_pes", got_npes, world),
        ("remote read", host.tolist(), expected.cpu().tolist()),
        ("signal inc sum", slot0, world),
        ("signal add sum", slot1, world * (world + 1) // 2),
    ]
    print(f"[rank {rank}] ---- results ----")
    ok = True
    for name, got, want in checks:
        passed = got == want
        ok &= passed
        print(f"[rank {rank}] {'PASS' if passed else 'FAIL'}  {name:16s} got={got} want={want}")

    tle.cleanup_communicator(data)
    tle.cleanup_communicator(sig)

    bad = torch.tensor([0 if ok else 1], device="npu", dtype=torch.int32)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if rank == 0:
        print("OVERALL:", "ALL PASS" if int(bad.cpu().item()) == 0 else "FAILURES")
    sys.exit(int(bad.cpu().item()))


if __name__ == "__main__":
    main()

"""SSD-streamed routed experts for the stacked mlx-community affine-4-bit layout.

``switch_mlp.fc1`` / ``fc2`` store all 128 experts stacked along the leading axis. A read fetches only the six
contiguous per-expert slices (weight, scales, biases for each projection) with ``pread``; the full stacked tensor is
never mapped. Exactly the router-selected experts are computed (top-6 of 128) and the caller adds the shared expert,
so no expert is skipped or approximated.

Speed (measured on this machine: one reader thread 2.8 GB/s, four threads 8.3 GB/s, uncached):
* the slices of every needed expert of a layer are read **in parallel** by a small thread pool;
* **speculative prefetch**: while layer L runs, the router of the next MoE layer is applied to L's input to predict
  its experts, and those slices are read in the background into a bounded host staging area. A wrong prediction only
  wastes bandwidth: the real router decides which experts are computed, so results are unchanged;
* decode needs a single GPU sync per layer (no per-expert evaluation), and results are assembled with one gather;
* the resident pool is a byte-bounded LRU of ready-to-use arrays shared by all layers.

Residency is bounded by the pool plus the staging area plus the experts of the layer in flight; activations and the
MLX allocator cache are budgeted by the admission module.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .runtime import Cancelled

_NP = {'U32': np.uint32, 'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16}
_PARTS = ('weight', 'scales', 'biases')
_STACK_NAMES = ('fc1.weight', 'fc1.scales', 'fc1.biases', 'fc2.weight', 'fc2.scales', 'fc2.biases')
PREDICT_EXTRA = int(os.environ.get('VEYSTRIX_NEMOTRON_PREDICT_EXTRA', '4'))   # predicted beyond top-k (spare bandwidth)
# Opt-in: the stock gather_qmm kernel over the stacked selected experts. Bit-identical to the stock mlx-lm layer on the
# synthetic checkpoints, but measured slower on the real model (stacking copies ~34 MB per layer): 4.4/5.6 vs 6.5 tok/s.
GATHER = os.environ.get('VEYSTRIX_NEMOTRON_GATHER', '0') == '1'
GATHER_MAX_EXPERTS = 24      # beyond this (large prefill chunks) the per-expert path avoids a large stacking copy
PREFILL_PREDICT_CAP = 40     # experts staged per layer during prefill (nearly all are needed there anyway)


class ExpertReadError(RuntimeError):
    """A shard changed, was truncated, or returned a short read."""


class StackedExpertReader:
    """Validated per-expert slice reads from stacked ``switch_mlp`` tensors. Thread safe."""

    def __init__(self, directory, index: dict, *, layers, expert_count: int, hidden: int,
                 moe_dim: int, group_size: int):
        self.directory = Path(directory)
        self.expert_count = expert_count
        self.group_size = group_size
        self._fds: dict[str, int] = {}
        self._sizes: dict[str, int] = {}
        self._lock = threading.Lock()
        self.bytes_read = 0
        self.reads = 0
        self.read_seconds = 0.0
        self._slices: dict[int, list] = {}
        for layer in layers:
            base = f'backbone.layers.{layer}.mixer.switch_mlp'
            entries = []
            for projection, rows, cols in (('fc1', moe_dim, hidden), ('fc2', hidden, moe_dim)):
                for part in _PARTS:
                    desc = index.get(f'{base}.{projection}.{part}')
                    if desc is None:
                        raise ExpertReadError(f'missing tensor {base}.{projection}.{part}')
                    last = cols // 8 if part == 'weight' else cols // group_size
                    if tuple(desc['shape']) != (expert_count, rows, last):
                        raise ExpertReadError(f'unexpected shape for {base}.{projection}.{part}')
                    per_expert = desc['length'] // expert_count
                    if per_expert * expert_count != desc['length'] or desc['dtype'] not in _NP:
                        raise ExpertReadError(f'unsupported stacked tensor {base}.{projection}.{part}')
                    entries.append((f'{projection}.{part}', desc['shard'], desc['offset'], per_expert,
                                    desc['dtype'], (rows, last)))
            self._slices[layer] = entries
        self.expert_bytes = sum(e[3] for e in next(iter(self._slices.values()))) if self._slices else 0

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds.clear()

    def _fd(self, shard: str):
        with self._lock:
            fd = self._fds.get(shard)
            if fd is None:
                fd = os.open(self.directory / shard, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
                self._fds[shard] = fd
                self._sizes[shard] = os.fstat(fd).st_size
            return fd, self._sizes[shard]

    def read_host(self, layer: int, expert: int) -> dict:
        """The six slices of one expert as numpy arrays (no MLX objects; safe to call from any thread)."""
        if layer not in self._slices or type(expert) is not int or not 0 <= expert < self.expert_count:
            raise ExpertReadError('invalid layer or expert id')
        began = time.perf_counter()
        out, total = {}, 0
        for desc in self._slices[layer]:
            name, array = self._pread_slice(desc, expert)
            total += array.nbytes
            out[name] = array
        with self._lock:
            self.bytes_read += total
            self.reads += 1
            self.read_seconds += time.perf_counter() - began
        return out

    def _pread_slice(self, desc, expert: int):
        name, shard, offset, length, dtype, shape = desc
        fd, size = self._fd(shard)
        start = offset + expert * length
        if start + length > size:
            raise ExpertReadError('shard truncated')
        if hasattr(os, 'preadv'):
            array = np.empty(shape, dtype=_NP[dtype])
            if os.preadv(fd, [memoryview(array).cast('B')], start) != length:
                raise ExpertReadError('short expert read')
        else:
            data = os.pread(fd, length, start)
            if len(data) != length:
                raise ExpertReadError('short expert read')
            array = np.frombuffer(data, dtype=_NP[dtype]).reshape(shape)
        return name, array

    def _read_projection(self, descriptors, expert: int):
        began = time.perf_counter()
        parts = dict(self._pread_slice(desc, expert) for desc in descriptors)
        # The MLX copy runs on the reader thread. The bfloat16 view runs
        # later on the thread that evaluates the arrays.
        values = {name: mx.array(array) for name, array in parts.items()}
        with self._lock:
            self.bytes_read += sum(array.nbytes for array in parts.values())
            self.read_seconds += time.perf_counter() - began
        return values

    def submit_expert(self, executor, layer: int, expert: int) -> cf.Future:
        """Queue both projections independently, batching each weight and its small metadata."""
        if layer not in self._slices or type(expert) is not int or not 0 <= expert < self.expert_count:
            raise ExpertReadError('invalid layer or expert id')
        parent = cf.Future()
        parent.set_running_or_notify_cancel()
        descriptors = self._slices[layer]
        projections = (descriptors[:len(_PARTS)], descriptors[len(_PARTS):])
        results, errors, remaining, gate = {}, [], [len(projections)], threading.Lock()

        def done(future):
            try:
                results.update(future.result())
            except BaseException as error:
                errors.append(error)
            with gate:
                remaining[0] -= 1
                last = remaining[0] == 0
            if last:
                with self._lock:
                    self.reads += 1
                if errors:
                    parent.set_exception(errors[0])
                else:
                    parent.set_result(results)

        for projection in projections:
            try:
                executor.submit(self._read_projection, projection, expert).add_done_callback(done)
            except RuntimeError as error:        # executor shut down
                errors.append(error)
                with gate:
                    remaining[0] -= 1
                    last = remaining[0] == 0
                if last:
                    parent.set_exception(errors[0])
        return parent

    def read(self, layer: int, expert: int) -> dict:
        return to_mlx(self.read_host(layer, expert), self._slices[layer])


def to_mlx(host: dict, entries=None) -> dict:
    """numpy slices -> MLX arrays (bfloat16 stored as uint16 is reinterpreted)."""
    out = {}
    for name, array in host.items():
        value = mx.array(array)
        out[name] = value.view(mx.bfloat16) if array.dtype == np.uint16 else value
    return out


class ExpertPool:
    """Resident LRU + parallel reads + staged prefetch, shared by every MoE layer."""

    def __init__(self, reader: StackedExpertReader, budget_bytes: int, *, workers: int = 8,
                 prefetch_workers: int = 6, staged_bytes: int = 384 * 2**20):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError('invalid expert pool budget')
        self.reader, self.budget = reader, budget_bytes
        self.staged_budget = staged_bytes
        self._items: OrderedDict = OrderedDict()      # (layer, expert) -> (mx arrays, size)   main thread only
        self._staged: OrderedDict = OrderedDict()     # (layer, expert) -> (mx arrays, size)   guarded by _lock
        self._inflight: dict = {}                     # (layer, expert) -> Future (prefetch)  guarded by _lock
        self._staged_total = 0
        self._lock = threading.Lock()
        self._demand = cf.ThreadPoolExecutor(workers, thread_name_prefix='nemotron-read')
        self._ahead = cf.ThreadPoolExecutor(prefetch_workers, thread_name_prefix='nemotron-ahead')
        self.resident_bytes = 0
        self.hits = self.misses = 0
        self.prefetch_issued = self.prefetch_used = self.prefetch_wasted = 0
        self.wait_seconds = 0.0
        self.cancel_check = lambda: False
        self.speculate = True        # False: layer-major prefill stages whole layers instead of predicting
        self._held = None            # layer-major prefill: experts of the layer in flight, released when it ends
        self._held_layer = None

    # -- layer scope (layer-major prefill) ---------------------------------------------------------------------
    def begin_layer(self, layer: int) -> None:
        """Keep every expert fetched for ``layer`` until ``end_layer``: the layer's token chunks then share one read.
        Held experts bypass the LRU so a long prefill does not evict what decode needs."""
        self._held, self._held_layer = {}, layer

    def end_layer(self) -> None:
        self._held, self._held_layer = None, None

    def held_bytes(self) -> int:
        held = self._held
        return sum(sum(a.nbytes for a in arrays.values()) for arrays in list(held.values())) if held else 0

    def prefetch_layer(self, layer: int) -> int:
        """Stage every expert of ``layer`` in the background (layer-major prefill: nearly all are needed).
        The staging area is raised to one full layer for the duration of the call's reads."""
        need = self.reader.expert_count * self.reader.expert_bytes + self.reader.expert_bytes
        self.staged_budget = max(self.staged_budget, need)
        return self.prefetch(layer, range(self.reader.expert_count))

    # -- prefetch ----------------------------------------------------------------------------------------------
    def prefetch(self, layer: int, experts) -> int:
        """Start background reads for experts not already resident, staged or in flight. Never blocks."""
        issued = 0
        for expert in experts:
            key = (layer, int(expert))
            with self._lock:
                if key in self._items or key in self._staged or key in self._inflight:
                    continue
                if self._staged_total + len(self._inflight) * self.reader.expert_bytes >= self.staged_budget:
                    break
                future = self.reader.submit_expert(self._ahead, key[0], key[1])
                self._inflight[key] = future
            future.add_done_callback(lambda f, key=key: self._landed(key, f))
            issued += 1
        self.prefetch_issued += issued
        return issued

    def _landed(self, key, future) -> None:
        try:
            host = future.result()
        except BaseException:
            with self._lock:
                self._inflight.pop(key, None)
            return
        size = sum(a.nbytes for a in host.values())
        with self._lock:
            if self._inflight.pop(key, None) is None:      # consumed by a demand fetch while reading
                return
            self._staged[key] = (host, size)
            self._staged_total += size
            while self._staged_total > self.staged_budget and self._staged:
                _, (_, old) = self._staged.popitem(last=False)
                self._staged_total -= old
                self.prefetch_wasted += 1

    # -- demand ------------------------------------------------------------------------------------------------
    def fetch(self, layer: int, experts, *, after_submit=None) -> dict:
        """Arrays for every requested expert; missing ones are read in parallel. Returns {expert: arrays}."""
        if self.cancel_check():
            raise Cancelled()
        out, waiting, direct = {}, {}, {}
        held = self._held if self._held_layer == layer else None
        for expert in experts:
            expert = int(expert)
            key = (layer, expert)
            if held is not None and expert in held:
                self.hits += 1
                out[expert] = held[expert]
                continue
            if key in self._items:
                self.hits += 1
                self._items.move_to_end(key)
                out[expert] = self._items[key][0]
                continue
            with self._lock:
                staged = self._staged.pop(key, None)
                future = self._inflight.get(key) if staged is None else None
                if staged is not None:
                    self._staged_total -= staged[1]
                elif future is not None:
                    del self._inflight[key]                  # we take ownership of the result
            if staged is not None:
                self.prefetch_used += 1
                out[expert] = self._admit(key, staged[0])
                self.hits += 1
            elif future is not None:
                waiting[expert] = future
            else:
                direct[expert] = self.reader.submit_expert(self._demand, layer, expert)
        if after_submit is not None:
            after_submit()
        began = time.perf_counter()
        for expert, future in waiting.items():
            self.prefetch_used += 1
            self.hits += 1
            out[expert] = self._admit((layer, expert), self._result(future))
        for expert, future in direct.items():
            self.misses += 1
            out[expert] = self._admit((layer, expert), self._result(future))
        self.wait_seconds += time.perf_counter() - began
        return out

    def _result(self, future):
        while True:
            try:
                return future.result(timeout=0.05)
            except cf.TimeoutError:
                if self.cancel_check():
                    raise Cancelled()

    def _admit(self, key, arrays):
        arrays = {n: (a.view(mx.bfloat16) if a.dtype == mx.uint16 else a) for n, a in arrays.items()}
        size = sum(a.nbytes for a in arrays.values())
        if self._held is not None and key[0] == self._held_layer:
            self._held[key[1]] = arrays
            return arrays
        if size <= self.budget:
            while self._items and self.resident_bytes + size > self.budget:
                _, (_, old) = self._items.popitem(last=False)
                self.resident_bytes -= old
            self._items[key] = (arrays, size)
            self.resident_bytes += size
        return arrays

    def get(self, layer: int, expert: int) -> dict:
        if self.cancel_check():
            raise Cancelled()
        return self.fetch(layer, [expert])[int(expert)]

    def clear(self, *, staged: bool = True) -> None:
        self._items.clear()
        self.resident_bytes = 0
        if staged:
            with self._lock:
                self._staged.clear()
                self._staged_total = 0

    def close(self) -> None:
        self._ahead.shutdown(wait=False, cancel_futures=True)
        self._demand.shutdown(wait=False, cancel_futures=True)

    def stats(self) -> dict:
        return {'hits': self.hits, 'misses': self.misses, 'resident_bytes': self.resident_bytes,
                'budget_bytes': self.budget, 'bytes_read': self.reader.bytes_read,
                'prefetch_issued': self.prefetch_issued, 'prefetch_used': self.prefetch_used,
                'prefetch_wasted': self.prefetch_wasted, 'staged_bytes': self._staged_total,
                'wait_seconds': round(self.wait_seconds, 4), 'read_seconds': round(self.reader.read_seconds, 4),
                'held_bytes': self.held_bytes()}


class NextGate:
    """The next MoE layer's router, kept outside the module tree (so its weights are not registered twice)."""

    def __init__(self, layer, weight, bias, top_k):
        self.layer, self.weight, self.bias, self.top_k = layer, weight, bias, top_k


class StreamedSwitchMLP(nn.Module):
    """Drop-in for mlx-lm ``SwitchMLP`` (relu2 MLP): returns ``[..., K, hidden]``."""

    def __init__(self, pool: ExpertPool, layer: int, group_size: int, eval_every: int = 24):
        super().__init__()
        self.pool, self.layer, self.group_size, self.eval_every = pool, layer, group_size, eval_every
        self.next_gate = None

    def _prefetch_next(self, flat, rows_n):
        nxt = self.next_gate
        if nxt is None or not self.pool.speculate:
            return
        scores = mx.sigmoid((flat @ nxt.weight.T).astype(mx.float32)) + nxt.bias
        k = min(nxt.top_k + PREDICT_EXTRA, scores.shape[-1])
        picks = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        ids = np.asarray(picks).reshape(-1)
        if rows_n > 1:
            counts = np.bincount(ids, minlength=scores.shape[-1])
            ids = np.argsort(-counts)[:PREFILL_PREDICT_CAP]
            ids = ids[counts[ids] > 0]
        self.pool.prefetch(nxt.layer, [int(e) for e in ids])

    def _gather(self, flat, ids, unique, arrays, lead, dtype):
        """The stock mlx-lm ``SwitchMLP`` computation (``gather_qmm``) over just the selected experts: they are stacked
        in ``unique`` order and the router indices are remapped to positions in that stack. Same kernel and index
        semantics as the stock layer, one launch per projection instead of a matmul per expert."""
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
        order = [int(e) for e in unique]
        stacked = {n: mx.stack([arrays[e][n] for e in order]) for n in _STACK_NAMES}
        local = mx.array(np.searchsorted(unique, ids).astype(np.int32))
        x = mx.expand_dims(flat, (-2, -3))
        do_sort = local.size >= 64
        idx, inv = local, None
        if do_sort:
            x, idx, inv = _gather_sort(x, local)
        h = mx.gather_qmm(x, stacked['fc1.weight'], stacked['fc1.scales'], stacked['fc1.biases'], rhs_indices=idx,
                          transpose=True, group_size=self.group_size, bits=4, mode='affine', sorted_indices=do_sort)
        y = mx.gather_qmm(nn.relu2(h), stacked['fc2.weight'], stacked['fc2.scales'], stacked['fc2.biases'],
                          rhs_indices=idx, transpose=True, group_size=self.group_size, bits=4, mode='affine',
                          sorted_indices=do_sort)
        if do_sort:
            y = _scatter_unsort(y, inv, local.shape)
        return y.squeeze(-2).reshape(*lead, local.shape[-1], flat.shape[-1]).astype(dtype)

    def _expert(self, x, weights):
        h = mx.quantized_matmul(x, weights['fc1.weight'], weights['fc1.scales'], weights['fc1.biases'],
                                group_size=self.group_size, bits=4)
        return mx.quantized_matmul(nn.relu2(h), weights['fc2.weight'], weights['fc2.scales'], weights['fc2.biases'],
                                   group_size=self.group_size, bits=4)

    def __call__(self, x, indices):
        lead = x.shape[:-1]
        hidden = x.shape[-1]
        flat = x.reshape(-1, hidden)
        ids = np.asarray(indices).reshape(flat.shape[0], -1)
        rows_n, k = ids.shape
        unique = np.unique(ids)
        if rows_n == 1 and not GATHER:
            arrays = self.pool.fetch(self.layer, unique.tolist(), after_submit=lambda: self._prefetch_next(flat, rows_n))
            outputs = [self._expert(flat, arrays[int(expert)]) for expert in ids[0]]
            return mx.stack(outputs, axis=-2).reshape(*lead, k, hidden).astype(x.dtype)
        self._prefetch_next(flat, rows_n)
        arrays = self.pool.fetch(self.layer, unique.tolist())
        if GATHER and len(unique) <= GATHER_MAX_EXPERTS:
            return self._gather(flat, ids, unique, arrays, lead, x.dtype)
        ys, positions, batch = [], [], []
        for expert in unique:
            rows, slots = np.nonzero(ids == expert)
            w = arrays[int(expert)]
            sub = flat if len(rows) == rows_n else flat[mx.array(rows)]
            y = self._expert(sub, w)
            ys.append(y)
            positions.append(rows * k + slots)
            batch.append(y)
            if len(batch) >= self.eval_every:          # bound live expert weights during large prefill chunks
                mx.eval(*batch)
                batch = []
        stacked = ys[0] if len(ys) == 1 else mx.concatenate(ys, axis=0)
        order = np.argsort(np.concatenate(positions))
        out = stacked[mx.array(order)]
        return out.reshape(*lead, k, hidden).astype(x.dtype)

"""Bounded SSD record reader for experimental synthetic experts.

This is not a checkpoint-format adapter. Records must have been independently
validated by the caller; no real checkpoint layout is inferred here.
"""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path

class ExpertStore:
    def __init__(self, path, records, budget_bytes, max_record_bytes=64 * 1024 * 1024):
        self.path = Path(path)
        if not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError('invalid budget')
        if not isinstance(max_record_bytes, int) or max_record_bytes <= 0:
            raise ValueError('invalid record size limit')
        self.budget = budget_bytes
        self.max_record_bytes = max_record_bytes
        self.records = dict(records)  # key -> (offset, length, sha256)
        self.cache = OrderedDict()
        self.resident_bytes = 0
        self.hits = self.misses = self.bytes_read = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        offset, length, digest = self.records[key]
        if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length <= 0 or length > self.max_record_bytes:
            raise ValueError('invalid or oversized expert record')
        fd = os.open(self.path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            size = os.fstat(fd).st_size
            if offset + length > size:
                raise ValueError('expert record outside file')
            data = os.pread(fd, length, offset)
        finally:
            os.close(fd)
        self.bytes_read += len(data)
        if len(data) != length or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('damaged expert record')
        if length <= self.budget:
            while self.cache and self.resident_bytes + length > self.budget:
                _, old = self.cache.popitem(last=False)
                self.resident_bytes -= len(old)
            self.cache[key] = data
            self.resident_bytes += length
        return data

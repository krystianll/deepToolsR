"""Byte-budgeted numeric caches for a long-lived plotting session."""

from collections import OrderedDict
from collections.abc import MutableMapping
from dataclasses import fields, is_dataclass
import sys

import numpy as np


_EVICTION_ORDER = ('bitmaps', 'text', 'statistics', 'scan', 'layout')
_ENTRY_OVERHEAD = 256


def _children(item):
    if isinstance(item, np.ndarray):
        base = getattr(item, 'base', None)
        if base is not None:
            yield base
        if isinstance(item, np.ma.MaskedArray):
            # A masked result keeps separate data and mask array objects.
            yield item.data
            yield item._mask
        if item.dtype.hasobject:
            yield from item.flat
        if hasattr(item, '__dict__'):
            yield vars(item)
    elif isinstance(item, dict):
        for key, value in item.items():
            yield key
            yield value
    elif isinstance(item, (tuple, list, set, frozenset)):
        yield from item
    elif isinstance(item, memoryview):
        yield item.obj
    elif is_dataclass(item) and not isinstance(item, type):
        for field in fields(item):
            yield getattr(item, field.name)
    elif hasattr(item, '__dict__') and not isinstance(item, type):
        yield vars(item)


def _array_header(value):
    """Charge an ndarray object without charging its owned data twice."""
    size = sys.getsizeof(value)
    if value.flags.owndata:
        size -= value.nbytes
    return max(size, 0)


def _entry_cost(key, value):
    """Return object cost and unique retained array backing allocations."""
    seen = set()
    allocations = {}
    object_bytes = _ENTRY_OVERHEAD

    def visit(item):
        nonlocal object_bytes
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, np.ndarray):
            object_bytes += _array_header(item)
            if item.flags.owndata:
                address = item.__array_interface__['data'][0]
                allocation = (address, item.nbytes)
                allocations[allocation] = (item, item.nbytes)
        else:
            object_bytes += sys.getsizeof(item)
        for child in _children(item):
            visit(child)

    visit(key)
    visit(value)
    return object_bytes, allocations


class ByteBudgetCache:
    """Separate LRUs with a shared byte budget and unique backing charges."""

    KINDS = _EVICTION_ORDER

    def __init__(self, budget):
        self.budget = max(0, int(budget))
        self.bytes_used = 0
        self._entries = {kind: OrderedDict() for kind in self.KINDS}
        self._allocations = {}
        self.hits = {kind: 0 for kind in self.KINDS}
        self.misses = {kind: 0 for kind in self.KINDS}

    def _bucket(self, kind):
        try:
            return self._entries[kind]
        except KeyError as error:
            raise ValueError(f'unknown cache kind: {kind}') from error

    def get(self, kind, key):
        bucket = self._bucket(kind)
        if key not in bucket:
            self.misses[kind] += 1
            return None, False
        bucket.move_to_end(key)
        self.hits[kind] += 1
        return bucket[key][0], True

    def peek(self, kind, key):
        bucket = self._bucket(kind)
        return (bucket[key][0], True) if key in bucket else (None, False)

    def _drop(self, kind, key):
        _value, object_bytes, allocations = self._entries[kind].pop(key)
        self.bytes_used -= object_bytes
        for identity in allocations:
            item, size, count = self._allocations[identity]
            if count == 1:
                self.bytes_used -= size
                del self._allocations[identity]
            else:
                self._allocations[identity] = (item, size, count - 1)

    def put(self, kind, key, value):
        bucket = self._bucket(kind)
        if key in bucket:
            self._drop(kind, key)
        if not self.budget:
            return False
        object_bytes, allocations = _entry_cost(key, value)
        standalone = object_bytes + sum(size for _, size in
                                        allocations.values())
        if standalone > self.budget:
            return False
        bucket[key] = (value, object_bytes, tuple(allocations))
        self.bytes_used += object_bytes
        for identity, (item, size) in allocations.items():
            if identity in self._allocations:
                old_item, old_size, count = self._allocations[identity]
                self._allocations[identity] = (old_item, old_size, count + 1)
            else:
                self._allocations[identity] = (item, size, 1)
                self.bytes_used += size
        while self.bytes_used > self.budget:
            for victim_kind in self.KINDS:
                if self._entries[victim_kind]:
                    victim = next(iter(self._entries[victim_kind]))
                    self._drop(victim_kind, victim)
                    break
        return key in bucket

    def entry_cost(self, kind, key):
        value, hit = self.peek(kind, key)
        if not hit:
            return None
        object_bytes, allocations = _entry_cost(key, value)
        return object_bytes + sum(size for _, size in allocations.values())

    def clear(self):
        for kind in self.KINDS:
            self._entries[kind].clear()
        self._allocations.clear()
        self.bytes_used = 0

    def stats(self):
        return {'budget': self.budget, 'bytes_used': self.bytes_used,
                'entries': {kind: len(bucket) for kind, bucket in
                            self._entries.items()},
                'hits': dict(self.hits), 'misses': dict(self.misses)}


class TextExtentCache(MutableMapping):
    """The mapping interface expected by per-figure ``TextMeasurer``s."""

    def __init__(self, cache, *, on_hit=None, on_store=None):
        self.cache = cache
        self._transient = {}
        self._stored_this_request = set()
        self._on_hit = on_hit
        self._on_store = on_store

    def clear_transient(self):
        self._transient.clear()
        self._stored_this_request.clear()

    def __getitem__(self, key):
        value, hit = self.cache.get('text', key)
        if hit:
            if self._on_hit is not None and key not in self._stored_this_request:
                self._on_hit()
            return value
        return self._transient[key]

    def __setitem__(self, key, value):
        if self._on_store is not None:
            self._on_store()
        if self.cache.put('text', key, value):
            self._stored_this_request.add(key)
            self._transient.pop(key, None)
        else:
            self._transient[key] = value

    def __delitem__(self, key):
        if key in self._transient:
            del self._transient[key]
        else:
            self.cache._drop('text', key)

    def __iter__(self):
        return iter(set(self.cache._entries['text']) | set(self._transient))

    def __len__(self):
        return len(set(self.cache._entries['text']) | set(self._transient))

    def __contains__(self, key):
        return key in self._transient or self.cache.peek('text', key)[1]

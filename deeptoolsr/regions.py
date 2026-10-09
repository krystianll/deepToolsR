"""BED/GTF region files, read by the native ``_regions`` loader.

The loader reproduces deeptoolsintervals (deepTools 3.5.6): the same groups,
names, coordinates, exons and per-chromosome order. Plain and gzip files are
streamed natively; bzip2 files are decompressed here and streamed to it.
Blacklist and whitelist BED files are read by the same loader.
"""
import bz2
import os

from deeptoolsr import _regions
from deeptoolsr.native_path import native_path


def _read(target, path):
    path = os.fspath(path)
    with open(path, 'rb') as handle:
        compressed_bz2 = handle.read(3) == b'BZh'
    if compressed_bz2:
        with bz2.open(path, 'rb') as stream:
            target.read_stream(stream.read, path)
    else:
        target.read_file(native_path(path))


def load_regions(paths, transcript_id='transcript', exon_id='exon',
                 designator='transcript_id', keep_exons=False,
                 default_group=None):
    """Read one or more region files into one ``_regions.RegionSet``."""
    regions = _regions.RegionSet(transcript_id, exon_id, designator,
                                 keep_exons, default_group)
    for path in [paths] if isinstance(paths, str) else paths:
        _read(regions, path)
    regions.finish()
    return regions


def load_blacklist(path, merge_aliases=False):
    """Read a blacklist or whitelist BED into a ``_regions.Blacklist``;
    ``merge_aliases`` joins chr1/1 and chrM/MT entries under the first-seen
    name."""
    blacklist = _regions.Blacklist(merge_aliases)
    _read(blacklist, path)
    blacklist.finish()
    return blacklist


def blacklist_strand(region_strand, touch_strand):
    """The blacklist strand set ('.', '+' or '-') that applies to a region.

    ``sense``: entries on the region's strand; ``antisense``: on the opposite
    strand; ``ignore`` or a region without a +/- strand: every entry. Entries
    without a +/- strand belong to both strand sets.
    """
    if touch_strand == 'ignore' or region_strand not in ('+', '-'):
        return '.'
    if touch_strand == 'sense':
        return region_strand
    return '-' if region_strand == '+' else '+'

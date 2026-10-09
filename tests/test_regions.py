"""The native BED/GTF region loader and its parity with deeptoolsintervals."""
import bz2
import gzip
import os
from pathlib import Path

import pytest

from deeptoolsr.region_provenance import find_random_label
from deeptoolsr.regions import blacklist_strand, load_blacklist, load_regions

DATA = Path(__file__).parent / 'test_data'
SUBSET = DATA / 'regions' / 'gencode_v50_subset.gtf.gz'
WHOLE = 2 ** 32 - 2


def rows(regions, chrom, end=WHOLE):
    return [list(row) for row in regions.rows(chrom, end)]


def write(path, text, opener=open):
    with opener(path, 'wt', newline='') as handle:
        handle.write(text)
    return str(path)


def test_bed_groups_names_and_order(tmp_path, capsys):
    bed = write(tmp_path / 'a.bed',
                'chr1\t100\t200\tx\t0\t+\n'
                'chr1\t50\t300\tx\t.\t*\n'
                'chr1\t100\t150\tw\t2.5\t-\n'
                '#first\n'
                'chr1\t5\t5\tbad\t0\t+\n'
                'chr2\t-3\t9\ty\t1\t+')
    regions = load_regions(bed)
    assert regions.labels == ['first', 'a.bed']
    # findOverlaps order: start, end, name, group.
    assert rows(regions, 'chr1') == [
        ['chr1', [(50, 300)], 'x_r1', 0, '.', '.'],
        ['chr1', [(100, 150)], 'w', 0, '-', 2.5],
        ['chr1', [(100, 200)], 'x', 0, '+', 0.0]]
    assert rows(regions, 'chr2') == [['chr2', [(0, 9)], 'y', 1, '+', 1.0]]
    assert 'chr1:5-5 is an invalid BED interval' in capsys.readouterr().err


def test_empty_bed_group_is_dropped(tmp_path, capsys):
    bed = write(tmp_path / 'a.bed', 'chr1\t1\t9\n#A\n#B\nchr1\t2\t9\n')
    assert load_regions(bed, default_group='rest').labels == ['A', 'rest']
    assert "'B' group had no valid entries" in capsys.readouterr().err


def test_label_column_and_bed3_names(tmp_path):
    bed = write(tmp_path / 'l.bed',
                '#chrom\ts\te\tdeepTools_group\n'
                'chr1\t1\t9\tG2\nchr1\t1\t9\tG1\n# skipped\nchr1\t1\t9\tG2\n')
    regions = load_regions(bed)
    assert regions.labels == ['G2', 'G1']
    assert [row[2:4] for row in rows(regions, 'chr1')] == [
        ['chr1:1-9', 0], ['chr1:1-9', 1], ['chr1:1-9_r1', 0]]


def test_bed12_blocks(tmp_path, capsys):
    bed = write(tmp_path / 'b.bed',
                'chr1\t100\t500\tok\t0\t+\t100\t500\t0\t2\t20,10,\t50,0,\n'
                'chr1\t100\t500\tbad\t0\t+\t100\t500\t0\t2\t10,x\t0,50\n'
                'chr1\t100\t500\tfew\t0\t+\t100\t500\t0\t3\t10,20\t0,50\n')
    exons = {row[2]: row[1] for row in rows(load_regions(bed, keep_exons=True), 'chr1')}
    assert exons == {'ok': [(100, 110), (150, 170)], 'bad': [(100, 500)],
                     'few': [(100, 500)]}
    err = capsys.readouterr().err
    assert 'invalid exon offset' in err and 'too few exon' in err
    no_blocks = rows(load_regions(bed), 'chr1')
    assert all(row[1] == [(100, 500)] for row in no_blocks)


GTF = ('#!header\n'
       'chr1\tsrc\ttranscript\t11\t100\t.\t+\t.\tgene_id "g1"; transcript_id "t1";\n'
       'chr1\tsrc\texon\t11\t20\t.\t+\t.\tgene_id "g1"; transcript_id "t1";\n'
       'chr1\tsrc\tCDS\t15\t20\t.\t+\t.\tgene_id "g1"; transcript_id "t1";\n'
       'chr1\tsrc\texon\t51\t100\t.\t+\t.\tgene_id "g1"; transcript_id "t1";\n'
       'chr1\tsrc\ttranscript\t11\t100\t3\t-\t.\tgene_id "g1"; transcript_id "t1";\n'
       'MT\tsrc\tgene\t1\t9\t.\t-\t.\tgene_id "g2"; transcript_id "t2";\n'
       'MT\tsrc\ttranscript\t1\t9\t.\t-\t.\tgene_id "g2"; transcript_id "t2";')


def test_gtf_transcripts_exons_and_duplicates(tmp_path, capsys):
    gtf = write(tmp_path / 'a.gtf', GTF)
    regions = load_regions(gtf, keep_exons=True, default_group='genes')
    assert regions.labels == ['genes']
    assert rows(regions, 'chr1') == [
        ['chr1', [(10, 20), (50, 100)], 't1', 0, '+', '.']]
    assert 't1 occurs more than once' in capsys.readouterr().err
    cds = load_regions(gtf, exon_id='CDS', keep_exons=True)
    assert rows(cds, 'chr1')[0][1] == [(14, 20)]
    genes = load_regions(gtf, transcript_id='gene', designator='gene_id')
    assert rows(genes, 'chrM') == [['chrM', [(0, 9)], 'g2', 0, '-', '.']]


def test_chromosome_munging(tmp_path):
    regions = load_regions(write(tmp_path / 'a.bed',
                                 'chr1\t1\t9\n1\t2\t9\nMT\t1\t9\nchrM\t2\t9\n'))
    assert regions.chroms == ['chr1', 'MT']
    assert [regions.munge(c) for c in ('1', 'chrM', 'chr1', 'X')] == ['chr1', 'MT', 'chr1', 'X']
    assert len(rows(regions, '1')) == 2 and rows(regions, '1')[0][0] == '1'


def test_compressed_crlf_and_missing_eol(tmp_path):
    text = 'chr1\t10\t20\r\n#g\r\nchr1\t5\t15'
    expected = rows(load_regions(write(tmp_path / 'p.bed', text)), 'chr1')
    assert expected == [['chr1', [(5, 15)], 'chr1:5-15', 1, '.', '.'],
                        ['chr1', [(10, 20)], 'chr1:10-20', 0, '.', '.']]
    for name, opener in (('z.bed.gz', gzip.open), ('z.bed.bz2', bz2.open)):
        regions = load_regions(write(tmp_path / name, text, opener))
        assert rows(regions, 'chr1') == expected


def test_rows_end_before_chromosome_end(tmp_path):
    regions = load_regions(write(tmp_path / 'r.bed', 'chr1\t0\t50\nchr1\t350\t400\nchr1\t600\t700\n'))
    assert [row[1][0][0] for row in rows(regions, 'chr1', 600)] == [0, 350]


def test_blacklist_merges_per_strand(tmp_path):
    # Nested and overlapping entries merge (deepTools 3.5.6 re-admitted 300-500 here).
    path = write(tmp_path / 'bl.bed',
                 'track name=bl\n'
                 'chr1\t100\t500\ta\t0\t+\n'
                 'chr1\t200\t300\tb\t0\t+\n'
                 'chr1\t450\t600\tc\t0\t-\n'
                 'chr1\t900\t1200\n'
                 'chr1\t1300\t1400\td\t0\t.\n'
                 'chr1\t5\t5\n'
                 '1\t0\t10\n')
    blacklist = load_blacklist(path)
    assert blacklist.chroms == ['chr1', '1']
    assert blacklist.intervals('chr1') == [(100, 600), (900, 1200), (1300, 1400)]
    assert blacklist.intervals('chr1', '+') == [(100, 500), (900, 1200), (1300, 1400)]
    assert blacklist.intervals('chr1', '-') == [(450, 600), (900, 1200), (1300, 1400)]
    # Lookups prefer the exact name, then the alias, as the regions do.
    assert blacklist.intervals('1') == [(0, 10)]
    assert blacklist.intervals('chrM') == []
    assert load_blacklist(write(tmp_path / 'c.bed', '1\t0\t10\n')).intervals('chr1') == [(0, 10)]
    # Entries sharing a base with the segments (half-open).
    assert blacklist.overlaps('chr1', [(0, 100), (600, 950), (1250, 1301)]) == [(900, 1200), (1300, 1400)]
    assert blacklist.overlaps('chr1', [(550, 560), (0, 1000)], '+') == [(100, 500), (900, 1200)]
    assert blacklist.overlaps('chr2', [(0, 10)]) == []


def test_blacklist_whitespace_and_compression(tmp_path):
    text = 'chr1 10  20 x 0 -\nchr1\t30\t40\n'
    for name, opener in (('w.bed', open), ('w.bed.gz', gzip.open), ('w.bed.bz2', bz2.open)):
        blacklist = load_blacklist(write(tmp_path / name, text, opener))
        assert blacklist.intervals('chr1', '+') == [(30, 40)]
        assert blacklist.intervals('chr1', '-') == [(10, 20), (30, 40)]
    with pytest.raises(RuntimeError, match='BED line with fewer than 3 columns in .*bad.bed: chr1'):
        load_blacklist(write(tmp_path / 'bad.bed', 'chr1\t10\n'))


@pytest.mark.parametrize('region,touch,expected', [
    ('+', 'sense', '+'), ('-', 'sense', '-'), ('.', 'sense', '.'),
    ('+', 'antisense', '-'), ('-', 'antisense', '+'), ('.', 'antisense', '.'),
    ('+', 'ignore', '.'), ('-', 'ignore', '.')])
def test_blacklist_strand(region, touch, expected):
    assert blacklist_strand(region, touch) == expected


def test_group_names_in_file_order(tmp_path):
    bed = write(tmp_path / 'a.bed', 'chr2\t1\t9\tb\t0\t+\nchr1\t1\t9\ta\t0\t+\n#g\nchr1\t1\t9\tb\t0\t+\n')
    assert load_regions(bed).group_names() == [['b', 'a'], ['b']]


def test_errors(tmp_path):
    with pytest.raises(RuntimeError, match='recognized file type'):
        load_regions(write(tmp_path / 'e.bed', '# only a header\n'))
    with pytest.raises(RuntimeError, match='had valid regions'):
        load_regions(write(tmp_path / 'i.bed', 'chr1\t9\t1\n'))
    with pytest.raises(ValueError, match=r'x\.bed: invalid literal') as raised:
        load_regions(write(tmp_path / 'x.bed', 'chr1\t' + 'x' * 4096 + '\t110\tbad\t0\t+\n'))
    assert len(str(raised.value)) < 300 + len(str(tmp_path))


def test_find_random_label():
    assert find_random_label({'a', 'a_r1'}, 'a') == 'a_r2'
    assert find_random_label([], 'a') == 'a'


# --- parity with deeptoolsintervals (test-only reference) -------------------

def reference(paths, **options):
    GTF = pytest.importorskip('deeptoolsintervals').GTF
    gtf = GTF(paths, **options)
    chroms = list(dict.fromkeys(gtf.chroms))
    return gtf.labels, chroms, {
        chrom: [[chrom, o[4], o[2], o[3], o[5], o[6]] for o in gtf.findOverlaps(
            chrom, 0, WHOLE, trimOverlap=True, numericGroups=True, includeStrand=True) or []]
        for chrom in chroms}


NAMES = dict(transcriptID='transcript_id', exonID='exon_id', keepExons='keep_exons',
             transcript_id_designator='designator', defaultGroup='default_group')


def assert_parity(paths, **options):
    labels, chroms, expected = reference(paths, **options)
    regions = load_regions(paths, **{NAMES[k]: v for k, v in options.items()})
    assert (regions.labels, regions.chroms) == (labels, chroms)
    for chrom in chroms:
        # repr() compares NaN scores too.
        assert repr(rows(regions, chrom)) == repr(expected[chrom]), chrom


def reference_files():
    intervals = pytest.importorskip('deeptoolsintervals')
    return Path(intervals.__file__).parent / 'test'


@pytest.mark.parametrize('keep_exons', [False, True])
@pytest.mark.parametrize('name', ['GRCh38.84.gtf.gz', 'GRCh38.84.2.gtf.gz', 'GRCh38.84.bed',
                                  'GRCh38.84.bed2', 'GRCh38.84.bed6', 'GRCh38.84.bed12.bz2',
                                  'GRCh38.84.labels.bed', 'noOverlaps.bed', 'strands.bed'])
def test_parity_reference_files(name, keep_exons):
    assert_parity([str(reference_files() / name)], keepExons=keep_exons)


@pytest.mark.parametrize('names', [
    ['GRCh38.84.labels.bed', 'GRCh38.84.bed2'], ['GRCh38.84.bed2', 'GRCh38.84.labels.bed'],
    ['GRCh38.84.gtf.gz', 'GRCh38.84.2.gtf.gz'], ['GRCh38.84.gtf.gz', 'GRCh38.84.bed'],
    ['GRCh38.84.bed', 'GRCh38.84.gtf.gz'], ['GRCh38.84.bed6', 'GRCh38.84.bed12.bz2']])
def test_parity_multiple_files(names):
    assert_parity([str(reference_files() / name) for name in names], keepExons=True)


GTF_OPTIONS = [dict(), dict(keepExons=True, defaultGroup='genes'),
               dict(keepExons=True, exonID='CDS'),
               dict(keepExons=True, transcriptID='gene', transcript_id_designator='gene_id'),
               dict(transcript_id_designator='gene_name')]


@pytest.mark.parametrize('options', GTF_OPTIONS)
def test_parity_gencode_subset(tmp_path, options):
    assert_parity([str(SUBSET)], **options)
    # Ensembl naming: no 'chr' prefix, MT for chrM.
    ensembl = tmp_path / 'ensembl.gtf'
    with gzip.open(SUBSET, 'rt') as source, open(ensembl, 'w') as target:
        for line in source:
            target.write(line if line.startswith('#') else
                         line.replace('chrM\t', 'MT\t', 1).removeprefix('chr'))
    assert_parity([str(ensembl)], **options)


@pytest.mark.parametrize('name', ['test.gtf', 'test.bed3', 'test_filtering.blacklist.bed'])
def test_parity_repository_data(name):
    assert_parity([str(DATA / name)], keepExons=True, defaultGroup='genes')


def test_parity_irregular_bed(tmp_path):
    bed = write(tmp_path / 'odd.bed',
                'track name=x\n#hdr\n'
                'chr1\t100\t200\ta\t0\t+\r\nchr1\t100\t200\ta\t5\t-\n\n1\t50\t60\tb\t.\t*\n'
                'chr1\t-5\t20\tneg\t1.5\t+\nchr1\t30\t30\tzero\t0\t+\n#grpA\n'
                'chrM\t1\t10\tm\t0\t.\nMT\t5\t10\tm\t0\t.\nchr2\t5\t10\tx\tabc\t+\n#\n'
                'chr2\t1\t9\ty\t1e3\t-\nchr2\t1\t9\ty\tnan\t-')
    blocks = write(tmp_path / 'b.bed12.gz',
                   'chr1\t100\t500\tt\t0\t+\t1\t1\t0\t2\t50,50,\t150,0,\r\n'
                   'chr1\t100\t500\tu\t0\t+\t1\t1\t0\t2\t1,x\t0,1\n', gzip.open)
    assert_parity([bed, blocks], keepExons=True)


def test_parity_irregular_gtf(tmp_path):
    with gzip.open(SUBSET, 'rt') as handle:
        data = [line.rstrip('\n') for line in handle if not line.startswith('#')][:300]
    edits = {5: (3, '0'), 9: (8, 'gene_id "x"; tx_id "y";'), 11: (6, '?'), 13: (5, '7.5')}
    lines = []
    for index, line in enumerate(data):
        cols = line.split('\t')
        if index in edits:
            cols[edits[index][0]] = edits[index][1]
        if index % 100 == 0:
            cols[8] += ' deepTools_group "G{}";'.format(index // 100)
        lines.append('\t'.join(cols))
    lines += [line for line in data if '\ttranscript\t' in line][:2]
    plain = write(tmp_path / 'odd.gtf', '\n'.join(lines))
    crlf = write(tmp_path / 'crlf.gtf.gz', '\r\n'.join(lines) + '\r\n', gzip.open)
    for options in GTF_OPTIONS:
        assert_parity([plain], **options)
        assert_parity([crlf, plain], **options)


def test_parity_quoted_attributes(tmp_path):
    # csv quoting rules in the attribute column: quoted spaces and ';',
    # doubled, empty and unquoted-embedded quotes.
    attributes = ['transcript_id "a b;c"; deepTools_group "grp one";',
                  'transcript_id "q""x";',
                  'transcript_id ab"c;', 'transcript_id "t5"x;',
                  'transcript_id "";', 'transcript_id  "t7" ;']
    lines = []
    for index, attribute in enumerate(attributes):
        start = 100 * (index + 1)
        for feature, end in (('transcript', start + 90), ('exon', start + 20)):
            lines.append('chr1\tsrc\t{0}\t{1}\t{2}\t.\t+\t.\tgene_id "g{3}"; gene_name "n{3}"; {4}'
                         .format(feature, start, end, index, attribute))
    gtf = write(tmp_path / 'quoted.gtf', '\n'.join(lines) + '\n')
    for options in ({}, dict(keepExons=True),
                    dict(keepExons=True, transcript_id_designator='gene_name')):
        assert_parity([gtf], **options)


@pytest.mark.skipif(not os.environ.get('DEEPTOOLSR_GENCODE_GTF'),
                    reason='set DEEPTOOLSR_GENCODE_GTF to a full GENCODE GTF (slow)')
@pytest.mark.parametrize('keep_exons', [False, True])
def test_parity_full_gencode(keep_exons):
    assert_parity([os.environ['DEEPTOOLSR_GENCODE_GTF']], keepExons=keep_exons,
                  defaultGroup='genes')


def test_blacklist_gtf_uses_the_region_loader_transcripts(tmp_path):
    attributes = 'gene_id "g"; transcript_id "{}";'
    text = ''.join(f'chr1\tsrc\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t{attributes.format(name)}\n'
                   for feature, start, end, strand, name in [
                       ('gene', 1, 1000, '+', 't0'), ('transcript', 101, 200, '+', 't1'),
                       ('exon', 101, 150, '+', 't1'), ('transcript', 301, 400, '-', 't2'),
                       ('transcript', 501, 600, '.', 't3')])
    for name, opener in (('b.gtf', open), ('b.gtf.gz', gzip.open)):
        blacklist = load_blacklist(write(tmp_path / name, text, opener))
        assert blacklist.intervals('chr1') == [(100, 200), (300, 400), (500, 600)]
        assert blacklist.intervals('chr1', '+') == [(100, 200), (500, 600)]
        assert blacklist.intervals('chr1', '-') == [(300, 400), (500, 600)]
        assert blacklist.intervals('chr1') == [
            tuple(row[1][0]) for row in rows(load_regions(write(tmp_path / 'r.gtf', text)), 'chr1')]


def test_empty_first_names(tmp_path):
    # An empty name stored first (empty BED name column, GTF transcript_id "").
    bed = write(tmp_path / 'e.bed', 'chr1\t0\t10\t\t0\t+\nchr1\t20\t30\tb\t0\t+\n')
    gtf = write(tmp_path / 'e.gtf', 'chr1\tsrc\ttranscript\t1\t10\t.\t+\t.\tgene_id "g"; transcript_id "";\n'
                                    'chr1\tsrc\ttranscript\t21\t30\t.\t+\t.\tgene_id "g"; transcript_id "b";\n')
    for path in (bed, gtf):
        assert_parity([path])
        assert [row[2] for row in rows(load_regions(path), 'chr1')] == ['', 'b']
        assert load_blacklist(path).intervals('chr1') == [(0, 10), (20, 30)]


def test_gtf_three_column_lines(tmp_path):
    record = 'chr1\tsrc\ttranscript\t1\t10\t.\t+\t.\tgene_id "g"; transcript_id "t1";\n'
    # A transcript line without coordinates is skipped silently, as by the
    # reference, even after a line whose fourth column was not a number.
    for text in (record + 'chr1\tsrc\ttranscript\n',
                 record + 'chr1\tsrc\tgene\tabc\t10\t.\t+\t.\tgene_id "g";\nchr1\tsrc\ttranscript\n'):
        gtf = write(tmp_path / 't.gtf', text)
        assert_parity([gtf])
        assert rows(load_regions(gtf), 'chr1') == [['chr1', [(0, 10)], 't1', 0, '+', '.']]
    with pytest.raises(RuntimeError, match='Malformed GTF line'):
        load_regions(write(tmp_path / 'x.gtf', record + 'chr1\tsrc\texon\n'), keep_exons=True)


def test_truncated_gzip_is_an_error(tmp_path):
    data = gzip.compress(b''.join(b'chr1\t%d\t%d\n' % (i, i + 5) for i in range(0, 50000, 10)))
    for name, payload in (('trailer.bed.gz', data[:-8]), ('cut.bed.gz', data[:len(data) // 2])):
        path = tmp_path / name
        path.write_bytes(payload)
        for load in (load_regions, load_blacklist):
            with pytest.raises(RuntimeError, match='truncated gzip file'):
                load(str(path))


def test_blacklist_merges_chromosome_aliases_on_request(tmp_path):
    path = write(tmp_path / 'a.bed', 'chr1\t100\t200\n1\t300\t400\nMT\t0\t5\nchrM\t10\t20\n')
    merged = load_blacklist(path, merge_aliases=True)
    assert merged.chroms == ['chr1', 'MT']
    assert merged.intervals('chr1') == merged.intervals('1') == [(100, 200), (300, 400)]
    assert merged.intervals('chrM') == [(0, 5), (10, 20)]
    # bamCoverageR keeps every name as written.
    assert load_blacklist(path).chroms == ['chr1', '1', 'MT', 'chrM']


@pytest.mark.skipif(os.name != 'nt', reason='Windows path separators')
def test_windows_paths_give_basename_labels(tmp_path):
    bed = write(tmp_path / 'w.bed', 'chr1\t0\t10\ta\t0\t+\n')
    assert load_regions(bed.replace('/', '\\')).labels == ['w.bed']

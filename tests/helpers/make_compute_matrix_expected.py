"""Generate the pinned computeMatrix byte fixtures from small test inputs.

Run from the repository root with ``python -m tests.helpers.make_compute_matrix_expected``.
The output directory is fixed, but the same cases can be rendered into a
pytest temporary directory by calling :func:`run_case`.
"""

from pathlib import Path

from tests.helpers.bigwig import pyBigWig

from deeptoolsr import computeMatrix


EXPECTED = Path('tests/test_data/compute_matrix_expected')
INPUTS = EXPECTED / 'inputs'
HEATMAPPER = Path('tests/test_heatmapper')

CASES = {
    'reference_point': [
        'reference-point', '-S', str(HEATMAPPER / 'test.bw'),
        '-R', str(HEATMAPPER / 'test2.bed'), '-b', '100', '-a', '100',
        '-bs', '10'],
    'scale_regions': [
        'scale-regions', '-S', str(HEATMAPPER / 'test.bw'),
        '-R', str(HEATMAPPER / 'test2.bed'), '-b', '100', '-a', '100',
        '-m', '100', '-bs', '10'],
    'gtf': [
        'scale-regions', '-S', 'tests/test_data/test1.bw.bw',
        '-R', 'tests/test_data/test.gtf', '-b', '300', '-a', '500',
        '-m', '100', '-bs', '10'],
    'multiple_bed': [
        'reference-point', '-S', str(HEATMAPPER / 'test.bw'),
        '-R', str(HEATMAPPER / 'group1.bed'),
        str(HEATMAPPER / 'group2.bed'), '-b', '100', '-a', '100',
        '-bs', '10'],
    'blacklist_skip_zeros': [
        'reference-point', '-S', str(HEATMAPPER / 'test.bw'),
        '-R', str(HEATMAPPER / 'test2.bed'), '-b', '100', '-a', '100',
        '-bs', '10', '--blackListFileName', str(INPUTS / 'blacklist.bed'),
        '--skipZeros'],
    'exact_chromosomes': [
        'reference-point', '-S', str(INPUTS / 'both_names.bw'),
        '-R', str(INPUTS / 'both_names.bed'), '-b', '0', '-a', '10',
        '-bs', '10'],
    'chromosome_alias': [
        'reference-point', '-S', str(INPUTS / 'chr_name.bw'),
        str(INPUTS / 'bare_name.bw'), '-R', str(INPUTS / 'chr_name.bed'),
        '-b', '0', '-a', '10', '-bs', '10'],
}


def make_inputs():
    INPUTS.mkdir(parents=True, exist_ok=True)
    (INPUTS / 'blacklist.bed').write_text('ch1\t100\t150\n')
    (INPUTS / 'both_names.bed').write_text(
        'chr1\t10\t20\texact_chr\t0\t+\n'
        '1\t10\t20\texact_bare\t0\t+\n')
    (INPUTS / 'chr_name.bed').write_text('chr1\t10\t20\talias\t0\t+\n')
    for name, chroms, values in (
            ('both_names.bw', ('chr1', '1'), (2., 7.)),
            ('chr_name.bw', ('chr1',), (2.,)),
            ('bare_name.bw', ('1',), (7.,))):
        with pyBigWig.open(str(INPUTS / name), 'w') as bw:
            bw.addHeader([(chrom, 100) for chrom in chroms])
            bw.addEntries(list(chroms), [0] * len(chroms),
                          ends=[100] * len(chroms), values=list(values))


def run_case(name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = output_dir / '{}.mat.gz'.format(name)
    plain = output_dir / '{}.mat'.format(name)
    values = output_dir / '{}.tab'.format(name)
    base = CASES[name] + ['-p', '1', '--quiet', '--sortRegions', 'keep']
    computeMatrix.main(base + ['-o', str(matrix)])
    computeMatrix.main(base + ['-o', str(plain)])
    if name == 'reference_point':
        computeMatrix.main(base + ['-o', str(output_dir / 'export.mat.gz'),
                                   '--outFileNameMatrix', str(values)])


if __name__ == '__main__':
    make_inputs()
    for case_name in CASES:
        run_case(case_name, EXPECTED)

"""Batch computeMatrix construction and region order helpers."""

import os
import sys
from copy import deepcopy

import numpy as np
from deeptoolsintervals import GTF

from deeptoolsr import _compute_matrix_native, _statistics, _transform
from deeptoolsr import stats as kernels
from deeptoolsr.matrix import MatrixHeader, OwnedMatrix
from deeptoolsr.region_provenance import ANTISENSE_SOURCES, expand_antisense_groups


def smartLabel(label):
    """Strip a path and its final extension, retaining dotfile names."""
    name = os.path.splitext(os.path.basename(label))[0]
    return name or os.path.basename(label)


def smartLabels(labels):
    return [smartLabel(label) for label in labels]


def remove_empty_rows(matrix, *, threads):
    """Apply --skipZeros to the owned construction buffer and group bounds."""
    keep = np.flatnonzero(kernels.reduce(
        matrix.values, 'nonzero', 1, threads=threads) > 0)
    bounds = [int(np.count_nonzero(keep < bound))
              for bound in matrix.header.group_boundaries]
    matrix.regions = [matrix.regions[index] for index in keep]
    matrix.values = matrix.values[keep, :]
    parameters = dict(matrix.header.parameters)
    parameters['group_boundaries'] = bounds
    matrix.header = MatrixHeader.from_parameters(parameters)
    return matrix


def _validate_compute_lengths(parameters):
    if parameters['body'] > 0 and \
            parameters['body'] % parameters['bin size'] > 0:
        exit("The --regionBodyLength has to be "
             "a multiple of --binSize.\nCurrently the "
             "values are {} {} for\nregionsBodyLength and "
             "binSize respectively\n".format(parameters['body'],
                                             parameters['bin size']))

    # the beforeRegionStartLength is extended such that
    # length is a multiple of binSize
    if parameters['downstream'] % parameters['bin size'] > 0:
        exit("Length of region after the body has to be "
             "a multiple of --binSize.\nCurrent value "
             "is {}\n".format(parameters['downstream']))

    if parameters['upstream'] % parameters['bin size'] > 0:
        exit("Length of region before the body has to be a multiple of "
             "--binSize\nCurrent value is {}\n".format(parameters['upstream']))

    if parameters['unscaled 5 prime'] % parameters['bin size'] > 0:
        exit("Length of the unscaled 5 prime region has to be a multiple of "
             "--binSize\nCurrent value is {}\n".format(parameters['unscaled 5 prime']))

    if parameters['unscaled 3 prime'] % parameters['bin size'] > 0:
        exit("Length of the unscaled 5 prime region has to be a multiple of "
             "--binSize\nCurrent value is {}\n".format(parameters['unscaled 3 prime']))

    if parameters['unscaled 5 prime'] + parameters['unscaled 3 prime'] > 0 and parameters['body'] == 0:
        exit('Unscaled 5- and 3-prime regions only make sense with the scale-regions subcommand.\n')


def _gtf_options(all_args):
    transcript_id = 'transcript'
    exon_id = 'exon'
    designator = 'transcript_id'
    keep_exons = False
    quiet = False
    if all_args is not None:
        all_args = vars(all_args)
        transcript_id = all_args.get('transcriptID', transcript_id)
        exon_id = all_args.get('exonID', exon_id)
        designator = all_args.get('transcript_id_designator', designator)
        keep_exons = all_args.get('keepExons', keep_exons)
        quiet = all_args.get('quiet', quiet)
    return all_args, transcript_id, exon_id, designator, keep_exons, quiet


def chopRegions(exonsInput, left=0, right=0):
    """
    exons is a list of (start, end) tuples. The goal is to chop these into
    separate lists of tuples, to take care or unscaled regions. "left" and
    "right" denote regions of a given size to exclude from the normal binning
    process (unscaled regions).

    This outputs three lists of (start, end) tuples:

    leftBins: 5' unscaled regions
    bodyBins: body bins for scaling
    rightBins: 3' unscaled regions

    In addition are two integers
    padLeft: Number of bases of padding on the left (due to not being able to fulfill "left")
    padRight: As above, but on the right side
    """
    leftBins = []
    rightBins = []
    padLeft = 0
    padRight = 0
    exons = deepcopy(exonsInput)
    while len(exons) > 0 and left > 0:
        width = exons[0][1] - exons[0][0]
        if width <= left:
            leftBins.append(exons[0])
            del exons[0]
            left -= width
        else:
            leftBins.append((exons[0][0], exons[0][0] + left))
            exons[0] = (exons[0][0] + left, exons[0][1])
            left = 0
    if left > 0:
        padLeft = left

    while len(exons) > 0 and right > 0:
        width = exons[-1][1] - exons[-1][0]
        if width <= right:
            rightBins.append(exons[-1])
            del exons[-1]
            right -= width
        else:
            rightBins.append((exons[-1][1] - right, exons[-1][1]))
            exons[-1] = (exons[-1][0], exons[-1][1] - right)
            right = 0
    if right > 0:
        padRight = right

    return leftBins, exons, rightBins[::-1], padLeft, padRight


def chopRegionsFromMiddle(exonsInput, left=0, right=0):
    """
    Like chopRegions(), above, but returns two lists of tuples on each side of
    the center point of the exons.

    The steps are as follow:

     1) Find the center point of the set of exons (e.g., [(0, 200), (300, 400), (800, 900)] would be centered at 200)
       * If a given exon spans the center point then the exon is split
     2) The given number of bases at the end of the left-of-center list are extracted
       * If the set of exons don't contain enough bases, then padLeft is incremented accordingly
     3) As above but for the right-of-center list
     4) A tuple of (#2, #3, pading on the left, and padding on the right) is returned
    """
    leftBins = []
    rightBins = []
    size = sum([x[1] - x[0] for x in exonsInput])
    middle = size // 2
    cumulativeSum = 0
    padLeft = 0
    padRight = 0
    exons = deepcopy(exonsInput)

    # Split exons in half
    for exon in exons:
        size = exon[1] - exon[0]
        if cumulativeSum >= middle:
            rightBins.append(exon)
        elif cumulativeSum + size < middle:
            leftBins.append(exon)
        else:
            # Don't add 0-width exonic bins!
            if exon[0] < exon[1] - cumulativeSum - size + middle:
                leftBins.append((exon[0], exon[1] - cumulativeSum - size + middle))
            if exon[1] - cumulativeSum - size + middle < exon[1]:
                rightBins.append((exon[1] - cumulativeSum - size + middle, exon[1]))
        cumulativeSum += size

    # Trim leftBins/adjust padLeft
    lSum = sum([x[1] - x[0] for x in leftBins])
    if lSum > left:
        lSum = 0
        for i, exon in enumerate(leftBins[::-1]):
            size = exon[1] - exon[0]
            if lSum + size > left:
                leftBins[-i - 1] = (exon[1] + lSum - left, exon[1])
                break
            lSum += size
            if lSum == left:
                break
        i += 1
        if i < len(leftBins):
            leftBins = leftBins[-i:]
    elif lSum < left:
        padLeft = left - lSum

    # Trim rightBins/adjust padRight
    rSum = sum([x[1] - x[0] for x in rightBins])
    if rSum > right:
        rSum = 0
        for i, exon in enumerate(rightBins):
            size = exon[1] - exon[0]
            if rSum + size > right:
                rightBins[i] = (exon[0], exon[1] - rSum - size + right)
                break
            rSum += size
            if rSum == right:
                break
        rightBins = rightBins[:i + 1]
    elif rSum < right:
        padRight = right - rSum

    return leftBins, rightBins, padLeft, padRight


def compute_sub_matrix_wrapper(args):
    return ComputeMatrixBuilder.compute_sub_matrix_worker(*args)


def _get_chrom_sizes(bigwig_files, chrom_sizes_reader=None):
    """Intersect source chromosomes, preferring exact names to aliases."""
    reader = chrom_sizes_reader or _compute_matrix_native.bigwig_chrom_sizes
    cache = {}
    for path in bigwig_files:
        if path not in cache:
            cache[path] = dict(reader(path))
    tables = [cache[path] for path in bigwig_files]
    if not tables:
        raise ValueError('at least one bigWig is required')
    matched = [set() for _ in tables]
    common = []
    for chrom, size in tables[0].items():
        alternate = _compute_matrix_native.chromosome_alias(chrom)
        resolved = [chrom if chrom in table else alternate for table in tables]
        if all(table.get(name) == size for table, name in zip(tables, resolved)):
            common.append((chrom, size))
            for names, name in zip(matched, resolved):
                names.add(name)
    non_common = {(chrom, size) for table, names in zip(tables, matched)
                  for chrom, size in table.items() if chrom not in names}
    if non_common:
        sys.stderr.write('\nThe following chromosome names or lengths did not '
                         'match between the bigwig files\nchromosome\tlength\n')
        for chrom, size in sorted(non_common):
            sys.stderr.write(f'{chrom:>15}\t{size:>10}\n')
    if not common:
        raise SystemExit('No common chromosomes found. Are the bigwig files '
                         'from the same species and same assemblies?')
    return sorted(common), non_common


def _subtract_blacklist(tree, chrom, chunk):
    overlaps = tree.findOverlaps(chrom, chunk[0], chunk[1])
    if not overlaps:
        return [chunk]
    output = []
    for overlap in overlaps:
        if chunk[1] <= chunk[0]:
            break
        if chunk[0] < overlap[0]:
            output.append([chunk[0], overlap[0]])
        chunk[0] = overlap[1]
    if chunk[0] < chunk[1]:
        output.append([chunk[0], chunk[1]])
    return output


def _load_regions(owner, score_files, parameters, chrom_sizes, bed_files,
                  blacklist_file, transcript_id, exon_id,
                  transcript_designator, keep_exons, verbose):
    """Compute the single native batch in chromosome and BED order.

    Returns ``(batch, labels)``; ``batch`` is ``None`` when no region
    overlaps a scored chromosome.
    """
    if verbose:
        print('genome partition size for multiprocessing: {}'.format(
            max(size for _, size in chrom_sizes)))
    default_group = 'genes' if len(bed_files) == 1 else None
    bed = GTF(bed_files, defaultGroup=default_group,
              transcriptID=transcript_id, exonID=exon_id,
              transcript_id_designator=transcript_designator,
              keepExons=keep_exons)
    blacklist = GTF(blacklist_file) if blacklist_file else None
    planned_chroms = {chrom for chrom, _ in chrom_sizes}
    regions = []
    first_task = None
    for chrom, size in chrom_sizes:
        if chrom not in bed.chroms:
            bed_chrom = bed.mungeChromosome(chrom, append=False)
            if bed_chrom != chrom and bed_chrom in planned_chroms:
                continue
        chunks = (_subtract_blacklist(blacklist, chrom, [0, size])
                  if blacklist is not None else [[0, size]])
        for start, end in chunks:
            overlapping = [[chrom, item[4], item[2], item[3], item[5], item[6]]
                           for item in bed.findOverlaps(
                               chrom, start, end, trimOverlap=True,
                               numericGroups=True, includeStrand=True)]
            if not overlapping:
                continue
            if first_task is None:
                first_task = (owner, chrom, start, end)
            regions.extend(overlapping)
    if first_task is None:
        return None, bed.labels
    task = (*first_task, score_files, parameters, regions)
    return compute_sub_matrix_wrapper(task), bed.labels


class ComputeMatrixBuilder:
    """Own only the mutable state needed while constructing a batch matrix."""

    def __init__(self):
        self.quiet = True
        self.parameters = None
        self.matrix = None
        self.threads = 1

    def get_num_individual_matrix_cols(self):
        params = self.parameters
        return ((params['downstream'] + params['upstream'] + params['body'] +
                 params['unscaled 5 prime'] + params['unscaled 3 prime']) //
                params['bin size'])

    def computeMatrix(self, score_file_list, regions_file, parameters,
                      blackListFileName=None, verbose=False, allArgs=None,
                      *, threads):
        """
        Splits into
        multiple cores the computation of the scores
        per bin for each region (defined by a hash '#'
        in the regions (BED/GFF) file.
        """
        self.threads = threads
        parameters['proc number'] = threads
        _validate_compute_lengths(parameters)

        (allArgs, transcriptID, exonID, transcript_id_designator,
         keepExons, self.quiet) = _gtf_options(allArgs)

        chromSizes, _ = _get_chrom_sizes(score_file_list)
        batch, labels = _load_regions(
            self, score_file_list, parameters, chromSizes, regions_file,
            blackListFileName, transcriptID, exonID,
            transcript_id_designator, keepExons, verbose)
        if parameters.get('antisense') == 'as_groups':
            labels, parameters[ANTISENSE_SOURCES] = expand_antisense_groups(labels)
        if batch is None or len(batch[1]) == 0:
            sys.stderr.write("\nERROR: Either the BED file does not contain any valid regions or there are none remaining after filtering.\n")
            exit(1)
        # The batch holds the C++-owned matrix, its regions in chromosome
        # order, and the number of regions lacking scores. Rows are then
        # ordered by group.
        matrix, regions, regions_no_score = batch
        batch = None
        foo = sorted(zip([x[3] for x in regions], range(len(regions)),
                         regions))
        sortIdx = [x[1] for x in foo]
        regions = [x[2] for x in foo]
        # Apply the output->source permutation in place with one row of
        # scratch storage, keeping the C++ allocation (the returned ndarray
        # owns its std::vector through a capsule).
        _statistics.permute_rows_inplace(matrix, 0, matrix.shape[0], sortIdx)

        # Borrow the numeric allocation and add only the Boolean invalid mask;
        # masked_invalid(copy=True) would duplicate the complete values array.
        # Native kernels read NaNs directly; no resident matrix-sized mask.

        if matrix.shape[0] != len(regions):
            raise ValueError("matrix length does not match regions length")

        if regions_no_score == len(regions):
            exit("\nERROR: None of the BED regions could be found in the bigWig"
                 "file.\nPlease check that the bigwig file is valid and "
                 "that the chromosome names between the BED file and "
                 "the bigWig file correspond to each other\n")

        if regions_no_score > len(regions) * 0.75:
            file_type = 'bigwig' if score_file_list[0].endswith(".bw") else "BAM"
            prcnt = 100 * float(regions_no_score) / len(regions)
            sys.stderr.write(
                "\n\nWarning: {0:.2f}% of regions are *not* associated\n"
                "to any score in the given {1} file. Check that the\n"
                "chromosome names from the BED file are consistent with\n"
                "the chromosome names in the given {2} file and that both\n"
                "files refer to the same species\n\n".format(prcnt,
                                                             file_type,
                                                             file_type))

        self.parameters = parameters

        numcols = matrix.shape[1]
        num_ind_cols = self.get_num_individual_matrix_cols()
        sample_boundaries = list(range(0, numcols + num_ind_cols, num_ind_cols))
        if allArgs is not None and allArgs['samplesLabel'] is not None:
            sample_labels = allArgs['samplesLabel']
        else:
            sample_labels = smartLabels(score_file_list)
        if parameters.get('antisense') == 'as_samples':
            sample_labels = [expanded_label
                             for label in sample_labels
                             for expanded_label in (label, label + '_antisense')]

        # Determine the group boundaries
        group_boundaries = []
        group_labels_filtered = []
        last_idx = -1
        for x in range(len(regions)):
            if regions[x][3] != last_idx:
                last_idx = regions[x][3]
                group_boundaries.append(x)
                group_labels_filtered.append(labels[last_idx])
        group_boundaries.append(len(regions))

        # check if a given group is too small. Groups that
        # are too small can't be plotted and an exception is thrown.
        group_len = np.diff(group_boundaries)
        if len(group_len) > 1:
            sum_len = sum(group_len)
            group_frac = [float(x) / sum_len for x in group_len]
            if min(group_frac) <= 0.002:
                sys.stderr.write(
                    "One of the groups defined in the bed file is "
                    "too small.\nGroups that are too small can't be plotted. "
                    "\n")

        parameters['sample_labels'] = sample_labels
        parameters['group_labels'] = group_labels_filtered
        parameters['sample_boundaries'] = sample_boundaries
        parameters['group_boundaries'] = group_boundaries
        self.matrix = OwnedMatrix.from_compute(parameters, matrix, regions)
        if parameters['skip zeros']:
            remove_empty_rows(self.matrix, threads=threads)
        return self.matrix

    @staticmethod
    def compute_sub_matrix_worker(self, chrom, start, end, score_file_list, parameters, regions):
        """
        Returns
        -------
        numpy matrix
            A numpy matrix that contains per each row the values found per each of the regions given
        """
        if parameters['verbose']:
            sys.stderr.write("Processing {}:{}-{}\n".format(chrom, start, end))

        minus_score_files = parameters.get('score files minus')
        native_paths = list(score_file_list)
        if minus_score_files:
            native_paths.extend(minus_score_files)

        antisense_mode = parameters.get('antisense', 'skip')
        unstranded_mode = parameters.get('unstranded', 'sum_strands')
        output_roles = ('coding', 'antisense') if antisense_mode == 'as_samples' else ('coding',)

        # In as_groups mode each annotated region is followed by a copy whose
        # strand is reversed. The Boolean is kept outside the public region
        # record so the matrix format remains unchanged.
        work_regions = []
        for region in regions:
            coding_region = list(region)
            if antisense_mode == 'as_groups':
                coding_region[3] = 2 * int(coding_region[3])
            work_regions.append((coding_region, False))
            if antisense_mode == 'as_groups':
                antisense_region = list(region)
                antisense_region[2] = '{}_antisense'.format(antisense_region[2])
                antisense_region[3] = 2 * int(antisense_region[3]) + 1
                # Keep the region's own strand (do not flip it): flipping here
                # previously swapped which end is treated as TSS vs TES and
                # reversed the bin order for the antisense copy. Which bigwig
                # strand is read is instead selected explicitly below via
                # `is_antisense_group`, independent of bin orientation.
                work_regions.append((antisense_region, True))

        # determine the number of matrix columns based on the lengths
        # given by the user, times the number of output samples
        matrix_cols = len(score_file_list) * len(output_roles) * \
            ((parameters['downstream'] +
              parameters['unscaled 5 prime'] + parameters['unscaled 3 prime'] +
              parameters['upstream'] + parameters['body']) //
             parameters['bin size'])

        # Python computation fills a conventional NumPy allocation. Native
        # computation adopts the C++ batch allocation after zones are planned,
        # avoiding simultaneous full-sized input and destination matrices.
        sub_matrix = None

        j = 0
        sub_regions = []
        regions_no_score = 0
        native_specs = []
        native_records = []

        def accept_coverage(coverage, transcript):
            """Apply row filters and copy an accepted row to final storage."""
            nonlocal j, regions_no_score
            feature_chrom = transcript[0]
            exons = transcript[1]
            feature_name = transcript[2]
            if coverage is None:
                regions_no_score += 1
                if not self.quiet:
                    sys.stderr.write(
                        "No data was found for region {0} {1}:{2}-{3}. "
                        "Skipping...\n".format(
                            feature_name, feature_chrom,
                            exons[0][0], exons[-1][1]))
                coverage = np.full(
                    matrix_cols,
                    0.0 if parameters['missing data as zero'] else np.nan)

            failed, _ = kernels.filter_rows(
                np.asarray(coverage).reshape(1, -1), 'perBin',
                -np.inf if parameters['min threshold'] is None else parameters['min threshold'],
                np.inf if parameters['max threshold'] is None else parameters['max threshold'],
                inclusive=True, threads=parameters.get('proc number', 1))
            if failed[0]:
                return
            if parameters['scale'] != 1:
                coverage = np.asarray(coverage, dtype=np.float32).reshape(1, -1)
                _transform.scale_scalar(coverage, parameters['scale'], 1)
                coverage = coverage[0]
            sub_matrix[j, :] = coverage
            sub_regions.append(transcript)
            j += 1

        for transcript, is_antisense_group in work_regions:
            feature_chrom = transcript[0]
            exons = transcript[1]
            feature_start = exons[0][0]
            feature_end = exons[-1][1]
            feature_name = transcript[2]
            feature_strand = transcript[4]
            # deeptoolsintervals serializes every non-+/- BED strand (for
            # example "*" or ".") as ".". Treat all such values as the
            # same unstranded state.
            is_unstranded = feature_strand not in ('+', '-')
            # Unstranded regions need a plotting orientation even when their
            # signal is the sum of both physical DNA strands.
            if is_unstranded and minus_score_files is not None:
                orientation_strand = '-' if unstranded_mode == 'as_minus' else '+'
            else:
                orientation_strand = feature_strand
            padLeft = 0
            padRight = 0
            padLeftNaN = 0
            padRightNaN = 0
            upstream = []
            downstream = []

            # get the body length
            body_length = np.sum([x[1] - x[0] for x in exons]) - parameters['unscaled 5 prime'] - parameters['unscaled 3 prime']

            # print some information
            if parameters['body'] > 0 and \
                    body_length < parameters['bin size']:
                if not self.quiet:
                    sys.stderr.write("A region that is shorter than the bin size (possibly only after accounting for unscaled regions) was found: "
                                     "({0}) {1} {2}:{3}:{4}. Skipping...\n".format((body_length - parameters['unscaled 5 prime'] - parameters['unscaled 3 prime']),
                                                                                   feature_name, feature_chrom,
                                                                                   feature_start, feature_end))
                coverage = np.zeros(matrix_cols)
                if not parameters['missing data as zero']:
                    coverage[:] = np.nan
                native_records.append((transcript, coverage, None))
                continue
            else:
                if orientation_strand == '-':
                    if parameters['downstream'] > 0:
                        upstream = [(feature_start - parameters['downstream'], feature_start)]
                    if parameters['upstream'] > 0:
                        downstream = [(feature_end, feature_end + parameters['upstream'])]
                    unscaled5prime, body, unscaled3prime, padLeft, padRight = chopRegions(exons, left=parameters['unscaled 3 prime'], right=parameters['unscaled 5 prime'])
                    # bins per zone
                    a = parameters['downstream'] // parameters['bin size']
                    b = parameters['unscaled 3 prime'] // parameters['bin size']
                    d = parameters['unscaled 5 prime'] // parameters['bin size']
                    e = parameters['upstream'] // parameters['bin size']
                else:
                    if parameters['upstream'] > 0:
                        upstream = [(feature_start - parameters['upstream'], feature_start)]
                    if parameters['downstream'] > 0:
                        downstream = [(feature_end, feature_end + parameters['downstream'])]
                    unscaled5prime, body, unscaled3prime, padLeft, padRight = chopRegions(exons, left=parameters['unscaled 5 prime'], right=parameters['unscaled 3 prime'])
                    a = parameters['upstream'] // parameters['bin size']
                    b = parameters['unscaled 5 prime'] // parameters['bin size']
                    d = parameters['unscaled 3 prime'] // parameters['bin size']
                    e = parameters['downstream'] // parameters['bin size']
                c = parameters['body'] // parameters['bin size']

                # build zones (each is a list of tuples)
                #  zone0: region before the region start,
                #  zone1: unscaled 5 prime region
                #  zone2: the body of the region
                #  zone3: unscaled 3 prime region
                #  zone4: the region from the end of the region downstream
                #  the format for each zone is: [(start, end), ...], number of bins
                # Note that for "reference-point", upstream/downstream will go
                # through the exons (if requested) and then possibly continue
                # on the other side (unless parameters['nan after end'] is true)
                if parameters['body'] > 0:
                    zones = [(upstream, a), (unscaled5prime, b), (body, c), (unscaled3prime, d), (downstream, e)]
                elif parameters['ref point'] == 'TES':  # around TES
                    if orientation_strand == '-':
                        downstream, body, unscaled3prime, padRight, _ = chopRegions(exons, left=parameters['upstream'])
                        if padRight > 0 and parameters['nan after end'] is True:
                            padRightNaN += padRight
                        elif padRight > 0:
                            downstream.append((downstream[-1][1], downstream[-1][1] + padRight))
                        padRight = 0
                    else:
                        unscale5prime, body, upstream, _, padLeft = chopRegions(exons, right=parameters['upstream'])
                        if padLeft > 0 and parameters['nan after end'] is True:
                            padLeftNaN += padLeft
                        elif padLeft > 0:
                            upstream.insert(0, (upstream[0][0] - padLeft, upstream[0][0]))
                        padLeft = 0
                    e = np.sum([x[1] - x[0] for x in downstream]) // parameters['bin size']
                    a = np.sum([x[1] - x[0] for x in upstream]) // parameters['bin size']
                    zones = [(upstream, a), (downstream, e)]
                elif parameters['ref point'] == 'center':  # at the region center
                    if orientation_strand == '-':
                        upstream, downstream, padLeft, padRight = chopRegionsFromMiddle(exons, left=parameters['downstream'], right=parameters['upstream'])
                    else:
                        upstream, downstream, padLeft, padRight = chopRegionsFromMiddle(exons, left=parameters['upstream'], right=parameters['downstream'])
                    if padLeft > 0 and parameters['nan after end'] is True:
                        padLeftNaN += padLeft
                    elif padLeft > 0:
                        if len(upstream) > 0:
                            upstream.insert(0, (upstream[0][0] - padLeft, upstream[0][0]))
                        else:
                            upstream = [(downstream[0][0] - padLeft, downstream[0][0])]
                    padLeft = 0
                    if padRight > 0 and parameters['nan after end'] is True:
                        padRightNaN += padRight
                    elif padRight > 0:
                        downstream.append((downstream[-1][1], downstream[-1][1] + padRight))
                    padRight = 0
                    a = np.sum([x[1] - x[0] for x in upstream]) // parameters['bin size']
                    e = np.sum([x[1] - x[0] for x in downstream]) // parameters['bin size']
                    # It's possible for a/e to be floats or 0 yet upstream/downstream isn't empty
                    if a < 1:
                        upstream = []
                        a = 0
                    if e < 1:
                        downstream = []
                        e = 0
                    zones = [(upstream, a), (downstream, e)]
                else:  # around TSS
                    if orientation_strand == '-':
                        unscale5prime, body, upstream, _, padLeft = chopRegions(exons, right=parameters['downstream'])
                        if padLeft > 0 and parameters['nan after end'] is True:
                            padLeftNaN += padLeft
                        elif padLeft > 0:
                            upstream.insert(0, (upstream[0][0] - padLeft, upstream[0][0]))
                        padLeft = 0
                    else:
                        downstream, body, unscaled3prime, padRight, _ = chopRegions(exons, left=parameters['downstream'])
                        if padRight > 0 and parameters['nan after end'] is True:
                            padRightNaN += padRight
                        elif padRight > 0:
                            downstream.append((downstream[-1][1], downstream[-1][1] + padRight))
                        padRight = 0
                    a = np.sum([x[1] - x[0] for x in upstream]) // parameters['bin size']
                    e = np.sum([x[1] - x[0] for x in downstream]) // parameters['bin size']
                    zones = [(upstream, a), (downstream, e)]

                foo = parameters['upstream']
                bar = parameters['downstream']
                if orientation_strand == '-':
                    foo, bar = bar, foo
                if padLeftNaN > 0:
                    expected = foo // parameters['bin size']
                    padLeftNaN = int(round(float(padLeftNaN) / parameters['bin size']))
                    if expected - padLeftNaN - a > 0:
                        padLeftNaN += 1
                if padRightNaN > 0:
                    expected = bar // parameters['bin size']
                    padRightNaN = int(round(float(padRightNaN) / parameters['bin size']))
                    if expected - padRightNaN - e > 0:
                        padRightNaN += 1

                coverage = []
                native_indices = []
                native_scales = []
                n_score_files = len(score_file_list)
                for score_idx in range(n_score_files):
                    for output_role in output_roles:
                        if not minus_score_files:
                            indices = [score_idx]
                            scales = [parameters.get('scale plus', 1)]
                        elif is_unstranded and unstranded_mode == 'sum_strands':
                            indices = [score_idx, n_score_files + score_idx]
                            scales = [parameters.get('scale plus', 1),
                                      parameters.get('scale minus', 1)]
                        else:
                            coding_uses_minus = orientation_strand == '-'
                            # An as_samples 'antisense' output role and an
                            # as_groups antisense-group row both read the
                            # opposite physical bigwig strand from the
                            # region's own orientation; only the source
                            # selection flips, not bin layout/order.
                            use_minus = (not coding_uses_minus
                                         if (output_role == 'antisense' or
                                             is_antisense_group)
                                         else coding_uses_minus)
                            indices = [n_score_files + score_idx
                                       if use_minus else score_idx]
                            scales = [parameters.get(
                                'scale minus' if use_minus else 'scale plus', 1)]
                        if output_role == 'antisense' or is_antisense_group:
                            scales = [parameters.get('scale antisense', 1) * value
                                      for value in scales]
                        native_indices.append(indices)
                        native_scales.append(scales)

                native_zones = [
                    ([(int(start), int(end)) for start, end in segments],
                     int(n_bins))
                    for segments, n_bins in zones]
                spec_index = len(native_specs)
                native_specs.append((
                    feature_chrom, native_zones, native_indices,
                    native_scales, orientation_strand == '-',
                    padLeftNaN, padRightNaN))
                native_records.append((transcript, None, spec_index))
                continue
            accept_coverage(coverage, transcript)

        if native_specs:
            chroms = [spec[0] for spec in native_specs]
            zones = [spec[1] for spec in native_specs]
            source_indices = [spec[2] for spec in native_specs]
            source_scales = [spec[3] for spec in native_specs]
            reverse = [spec[4] for spec in native_specs]
            pad_left = [spec[5] for spec in native_specs]
            pad_right = [spec[6] for spec in native_specs]
            native_batch = _compute_matrix_native.bin_bigwig_batch(
                native_paths, chroms, zones, source_indices, source_scales,
                reverse, pad_left, pad_right,
                parameters['bin avg type'],
                parameters['missing data as zero'],
                parameters['proc number'])
            if all(ready is None for _, ready, _ in native_records):
                # The common path: compact/filter directly inside the native
                # batch allocation. Source rows are visited in increasing
                # order, so forward compaction cannot overwrite unread rows.
                sub_matrix = native_batch
            else:
                native_destination = _compute_matrix_native.NativeMatrix(
                    len(native_records), matrix_cols)
                sub_matrix = np.asarray(native_destination)
            for transcript, ready, spec_index in native_records:
                coverage = (ready if spec_index is None else
                            native_batch[spec_index, :])
                accept_coverage(coverage, transcript)
        else:
            native_destination = _compute_matrix_native.NativeMatrix(
                len(native_records), matrix_cols)
            sub_matrix = np.asarray(native_destination)
            for transcript, ready, _ in native_records:
                accept_coverage(ready, transcript)

        # remove empty rows
        sub_matrix = sub_matrix[0:j, :]
        if len(sub_regions) != len(sub_matrix[:, 0]):
            sys.stderr.write("regions lengths do not match\n")
        return sub_matrix, sub_regions, regions_no_score


def save_matrix_values(matrix, layout, labels, path):
    """Write computeMatrix's historical four-significant-digit value table."""
    parameters = matrix.header.parameters
    group_lengths = np.diff(layout.group_bounds)
    sample_lengths = np.diff(matrix.header.sample_boundaries)
    info = [f'{label}:{length}' for label, length in
            zip(labels.groups, group_lengths)]
    with open(path, 'wb') as handle:
        handle.write('#{}\n'.format('\t'.join(info)).encode('utf-8'))
        handle.write(('#downstream:{}\tupstream:{}\tbody:{}\tbin size:{}\t'
                      'unscaled 5 prime:{}\tunscaled 3 prime:{}\n').format(
                          parameters['downstream'], parameters['upstream'],
                          parameters['body'], parameters['bin size'],
                          parameters.get('unscaled 5 prime', 0),
                          parameters.get('unscaled 3 prime', 0)).encode('utf-8'))
        info.extend(label for label, length in
                    zip(labels.samples, sample_lengths)
                    for _ in range(length))
        handle.write('{}\n'.format('\t'.join(info)).encode('utf-8'))
        if layout.rows is None:
            np.savetxt(handle, matrix.values, fmt='%.4g', delimiter='\t')
        else:
            for row in layout.rows:
                np.savetxt(handle, matrix.values[int(row):int(row) + 1],
                           fmt='%.4g', delimiter='\t')

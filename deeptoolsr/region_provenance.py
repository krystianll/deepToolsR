"""Small, serializable source mappings for generated matrix groups."""

from deeptoolsintervals.parse import findRandomLabel


ANTISENSE_SOURCES = 'antisense_group_sources'


def expand_antisense_groups(labels):
    """Reserve literal names before assigning unique generated group names."""
    reserved = set(labels)
    expanded, sources = [], {}
    for label in labels:
        generated = findRandomLabel(reserved, label + '_antisense')
        reserved.add(generated)
        expanded.extend((label, generated))
        sources[generated] = label
    return expanded, sources


def antisense_sources(parameters):
    """Read explicit provenance, with a conservative legacy-file fallback.

    A null source denotes conflicting provenance after combining groups; it
    must never be guessed. Consumers still prefer exact selector identities.
    """
    if ANTISENSE_SOURCES in parameters:
        sources = parameters[ANTISENSE_SOURCES]
        if not isinstance(sources, dict) or any(
                not isinstance(label, str) or
                (source is not None and not isinstance(source, str))
                for label, source in sources.items()):
            raise ValueError('Invalid antisense group source metadata')
        return sources
    if parameters.get('antisense') != 'as_groups':
        return {}
    labels = parameters.get('group_labels', [])
    if len(labels) != len(set(labels)):
        raise ValueError('Legacy antisense matrix has ambiguous group names; '
                         'recompute it to record source identities')
    return {label: label[:-len('_antisense')] for label in labels
            if label.endswith('_antisense')}


def combined_antisense_sources(headers, output_labels=None):
    """Combine provenance by label, preserving conflicts instead of guessing.

    output_labels optionally relabels the concatenated input groups. Subset
    and reorder need no remapping because the keys are output group names.
    """
    combined, relevant, offset = {}, set(), 0
    for header in headers:
        sources = antisense_sources(header)
        for old_label in header['group_labels']:
            label = (output_labels[offset] if output_labels is not None
                     else old_label)
            source = sources.get(old_label)
            if old_label in sources:
                relevant.add(label)
            if label in combined and combined[label] != source:
                combined[label] = None
            else:
                combined[label] = source
            offset += 1
    return {label: source for label, source in combined.items() if label in relevant}


def update_antisense_sources(target, headers, output_labels=None):
    """Update only matrices with generation provenance; leave other headers alone."""
    if any(ANTISENSE_SOURCES in header or header.get('antisense') == 'as_groups'
           for header in headers):
        target[ANTISENSE_SOURCES] = combined_antisense_sources(headers, output_labels)

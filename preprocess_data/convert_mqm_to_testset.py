import csv
import json
import os
from collections import defaultdict

INPUT_FILE = "/anonymous/path/QE/data/generalMT/generalMT2024/mqm_generalMT2024_ende.tsv"
OUTPUT_DIR = "/anonymous/path/QE/data/test_improvement"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "en-de-2024-test-mqm.json")


def clean_target(target):
    """Remove <v> and </v> markers from target text."""
    return target.replace('<v>', '').replace('</v>', '')


def extract_error_spans(target):
    """
    Extract error spans from target with <v>...</v> markers.
    Returns (clean_text, list_of_spans) where each span is (start_char, end_char) in clean_text.
    """
    clean_parts = []
    spans = []
    pos = 0
    clean_pos = 0

    while pos < len(target):
        if target[pos:pos+3] == '<v>':
            pos += 3
            error_start = clean_pos
            close_pos = target.find('</v>', pos)
            if close_pos == -1:
                # malformed, just copy rest
                rest = target[pos:]
                clean_parts.append(rest)
                clean_pos += len(rest)
                break
            error_content = target[pos:close_pos]
            clean_parts.append(error_content)
            clean_pos += len(error_content)
            end_tag_pos = close_pos + 4  # </v>
            spans.append((error_start, clean_pos))
            pos = end_tag_pos
        elif target[pos:pos+4] == '</v>':
            pos += 4
        else:
            clean_parts.append(target[pos])
            clean_pos += 1
            pos += 1

    return ''.join(clean_parts), spans


def build_word_tags(mt_text, error_spans):
    """
    Build word-level OK/BAD tags and tokenized mt_tok.
    Tokenize roughly like the reference format (split on whitespace, keep punctuation separate).
    """
    # Simple tokenizer: split on whitespace, for each token mark if its char range
    # overlaps with any error span
    tokens = []
    token_positions = []  # (start, end) for each token in mt_text

    i = 0
    while i < len(mt_text):
        # Skip whitespace
        if mt_text[i].isspace():
            i += 1
            continue
        start = i
        # Collect non-whitespace
        while i < len(mt_text) and not mt_text[i].isspace():
            i += 1
        token = mt_text[start:i]
        tokens.append(token)
        token_positions.append((start, i))

    # Build tags
    tags = []
    for tok_start, tok_end in token_positions:
        is_bad = False
        for err_start, err_end in error_spans:
            # Check overlap: token range intersects error span
            if tok_start < err_end and tok_end > err_start:
                is_bad = True
                break
        tags.append("BAD" if is_bad else "OK")

    # Build mt_tok (simple whitespace tokenization with <EOS>)
    mt_tok = " ".join(tokens) + " <EOS>"

    # Handle empty case
    if not tokens:
        mt_tok = "<EOS>"

    return " ".join(tags) if tags else "", mt_tok


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read and group data
    groups = defaultdict(list)
    fieldnames = None
    malformed = 0

    with open(INPUT_FILE, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        for row in reader:
            key = (row['globalSegId'], row['system'])
            groups[key].append(row)

    print(f"Total (segment, system) pairs: {len(groups)}")

    results = []
    no_error_count = 0
    error_count = 0
    skipped_no_span = 0
    skipped_malformed = 0

    for (seg_id, system), rows in groups.items():
        src = rows[0]['source']

        # All rows for the same (segment, system) have the same target minus <v> markers.
        # Pick any row's target, clean markers to get the base MT text.
        mt_clean = clean_target(rows[0]['target'])

        # Collect all error annotations
        error_annotations = []
        for r in rows:
            cat = r['category']
            sev = r['severity']
            if cat == 'No-error' and sev == 'No-error':
                continue

            # Check for malformed category (looks like target text)
            if len(cat) > 100 or '\t' in cat or '\n' in cat:
                skipped_malformed += 1
                continue

            target_row = r['target']
            has_markers = '<v>' in target_row

            if has_markers:
                # Extract span from markers. extract_error_spans returns positions
                # in the cleaned version of target_row, which is identical to mt_clean.
                _, spans = extract_error_spans(target_row)
                if spans:
                    for s, e in spans:
                        # Skip zero-length spans (empty <v></v> markers) — treat as segment-level
                        if e <= s:
                            s, e = -1, -1
                        error_annotations.append({
                            'start': s,
                            'end': e,
                            'category': cat,
                            'severity': sev.lower() if sev.lower() in ('major', 'minor') else 'minor',
                        })
                else:
                    skipped_no_span += 1
            else:
                # No markers: source issue, omission, or other segment-level error
                error_annotations.append({
                    'start': -1,
                    'end': -1,
                    'category': cat,
                    'severity': sev.lower() if sev.lower() in ('major', 'minor') else 'minor',
                })

        # Deduplicate error annotations
        seen_spans = set()
        unique_errors = []
        for e in error_annotations:
            key = (e['start'], e['end'], e['category'], e['severity'])
            if key not in seen_spans:
                seen_spans.add(key)
                unique_errors.append(e)

        error_annotations = unique_errors

        # Skip no-error samples — only need samples with errors for testing
        if not error_annotations:
            no_error_count += 1
            continue
        else:
            error_count += 1
            starts = []
            ends = []
            severities = []
            error_types = []

            for e in error_annotations:
                starts.append(e['start'])
                ends.append(e['end'])
                severities.append(e['severity'])
                error_types.append(e['category'])

            item = {
                "src": src,
                "mt": mt_clean,
                "starts": starts,
                "ends": ends,
                "severities": severities,
                "error_types": error_types,
            }

            # Build word-level tags
            error_spans_for_tags = [(s, e) for s, e in zip(starts, ends) if s != -1]
            tags, mt_tok = build_word_tags(mt_clean, error_spans_for_tags)
            item["tags"] = tags
            item["mt_tok"] = mt_tok

        results.append(item)

    print(f"  No-error items: {no_error_count}")
    print(f"  Error items: {error_count}")
    print(f"  Skipped (no span found): {skipped_no_span}")
    print(f"  Skipped (malformed): {skipped_malformed}")
    print(f"  Total output items: {len(results)}")

    # Write output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {OUTPUT_FILE}")

    # Print error type distribution in the test set
    type_counter = {}
    for item in results:
        for et in item.get('error_types', []):
            type_counter[et] = type_counter.get(et, 0) + 1
    print("\nError type distribution:")
    for et, cnt in sorted(type_counter.items(), key=lambda x: -x[1]):
        print(f"  {et}: {cnt}")


if __name__ == '__main__':
    main()

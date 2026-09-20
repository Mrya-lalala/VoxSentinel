"""Canonical audit of the recorded Kathbath/IndicSynth lineage; no network access."""
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

LANGUAGES = {'bn':'bengali','gu':'gujarati','hi':'hindi','kn':'kannada','ml':'malayalam',
             'mr':'marathi','or':'odia','od':'odia','pa':'punjabi','sa':'sanskrit',
             'ta':'tamil','te':'telugu','ur':'urdu','oriya':'odia'}
REF = re.compile(r'^(\d+)-(\d+)-([mf])$', re.I)


def numeric_id(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        n = Decimal(str(value).strip())
        return str(int(n)) if n.is_finite() and n == n.to_integral_value() and n >= 0 else None
    except (InvalidOperation, ValueError):
        return None


def reference(value):
    if not value:
        return None
    stem = Path(str(value).replace('\\', '/')).name.lower()
    stem = re.sub(r'\.(wav|m4a|flac|mp3|ogg)$', '', stem)
    match = REF.fullmatch(stem)
    if not match:
        return None
    record, speaker, gender = match.groups()
    return {'record': str(int(record)), 'speaker': str(int(speaker)), 'gender': gender,
            'canonical': f'{int(record)}-{int(speaker)}-{gender}'}


def audit_identity(train, val):
    speakers = {s: defaultdict(set) for s in ('train', 'val')}
    refs = {s: defaultdict(set) for s in ('train', 'val')}
    issues, absent = [], []
    coverage, genders, rates = Counter(), Counter(), Counter()
    detailed_genders = Counter()
    hashes = defaultdict(list)
    role_counts = Counter()
    for split, rows in (('train', train), ('val', val)):
        for row in rows:
            wid, ds = row['window_id'], row['dataset_id']
            lang = str(row.get('spoken_language') or '').strip().lower()
            lang = LANGUAGES.get(lang, lang)
            label = int(row['label'])
            coverage[(split, lang, label, row.get('generator') or 'genuine')] += 1
            rates[(split, label, row.get('original_audio', {}).get('sample_rate'))] += 1
            digest = row.get('prepared_audio', {}).get('sha256')
            if digest:
                hashes[digest].append({'split': split, 'window_id': wid})
            else:
                issues.append({'window_id': wid, 'reason': 'missing_prepared_hash'})
            if not lang:
                issues.append({'window_id': wid, 'reason': 'missing_language'})
            parent = row.get('parent_refs') or {}
            ids = row.get('speaker_ids') or {}
            if ds == 'kathbath':
                roles = [('speaker', ids.get('speaker'), row.get('source_file'))]
            elif ds == 'indicsynth':
                roles = [(role, ids.get(role), parent.get(role+'_reference')) for role in ('source','target')]
            else:
                issues.append({'window_id': wid, 'reason': 'unsupported_origin_namespace', 'dataset': ds})
                continue  # Never merge unrelated numeric ID namespaces.
            for role, sid_raw, ref_raw in roles:
                sid, parsed = numeric_id(sid_raw), reference(ref_raw)
                is_tts_absent = ds == 'indicsynth' and role == 'source' and parent.get('source_kind') == 'tts_target_only' and sid_raw is None and ref_raw is None and parent.get('source_parent_verification') == 'not_applicable_tts'
                if is_tts_absent:
                    absent.append({'window_id': wid, 'role': role, 'reason': 'not_applicable_tts'})
                    genders[(split, label, role, 'not_applicable')] += 1
                    detailed_genders[(split, lang, label, role, 'not_applicable')] += 1
                    continue
                role_counts[(split, role)] += 1
                gender = parsed['gender'] if parsed else 'unknown'
                genders[(split, label, role, gender)] += 1
                detailed_genders[(split, lang, label, role, gender)] += 1
                evidence_ok = True
                if ds == 'indicsynth':
                    ev = (parent.get('reference_evidence') or {}).get(role) or {}
                    evref = reference(ev.get('fname'))
                    evidence_ok = bool(parsed and evref and evref['canonical'] == parsed['canonical'] and parent.get(role+'_parent_verification') == 'verified_train' and str(ev.get('shard', '')).startswith(lang+'/train-') and ev.get('method') in ('scanned_inventory','full_train_shard_scan'))
                if sid is None or parsed is None or sid != parsed['speaker'] or not evidence_ok:
                    issues.append({'window_id': wid, 'role': role, 'reason': 'missing_malformed_or_unverified_identity', 'speaker': sid_raw, 'reference': ref_raw, 'evidence_ok': evidence_ok})
                    continue
                # Common origin namespace follows documented, verified Kathbath parents.
                speakers[split][role].add(('kathbath', lang, sid))
                refs[split][role].add(('kathbath', lang, parsed['canonical']))
                if ds == 'kathbath' and str(ids.get('gender','')).lower() != gender:
                    issues.append({'window_id': wid, 'reason': 'gender_metadata_disagrees_with_filename'})
    def intersections(groups):
        result = {}
        for role in ('speaker','source','target'):
            for other in ('speaker','source','target'):
                overlap = groups['train'][role] & groups['val'][other]
                result[f'train_{role}_vs_val_{other}'] = {'count':len(overlap),'examples':[list(x) for x in sorted(overlap)[:10]]}
        return result
    speaker_overlap, reference_overlap = intersections(speakers), intersections(refs)
    dupes = {h: xs for h,xs in hashes.items() if len(xs)>1}
    cross = {h:xs for h,xs in dupes.items() if len({x['split'] for x in xs})>1}
    within = {s:{h:[x for x in xs if x['split']==s] for h,xs in dupes.items() if sum(x['split']==s for x in xs)>1} for s in ('train','val')}
    def records(counter, names):
        return [{**dict(zip(names,k)), 'windows':v} for k,v in sorted(counter.items(), key=lambda x:str(x[0]))]
    return {
        'schema':'voxsentinel.canonical_identity_audit.v1',
        'counts':{'train':len(train),'val':len(val)},
        'speaker_keys':{s:len(set().union(*speakers[s].values())) for s in speakers},
        'reference_keys':{s:len(set().union(*refs[s].values())) for s in refs},
        'speaker_key_intersections':speaker_overlap,'reference_recording_intersections':reference_overlap,
        'unresolved_required_count':len(issues),'unresolved_required':issues,
        'expected_absent_source_count':len(absent),'expected_absent_source_examples':absent[:10],
        'prepared_duplicates_cross_split':cross,'prepared_duplicates_within_split':within,
        'coverage':records(coverage, ['split','language','label','generator']),
        'gender_coverage':records(genders,['split','label','role','gender']),
        'gender_by_language':records(detailed_genders,['split','language','label','role','gender']),
        'within_split_shared_genuine_synthetic':{s:{'speakers':len(speakers[s]['speaker'] & (speakers[s]['source'] | speakers[s]['target'])), 'references':len(refs[s]['speaker'] & (refs[s]['source'] | refs[s]['target']))} for s in ('train','val')},
        'original_sample_rates':records(rates,['split','label','sample_rate']),
        'ready':not issues and not cross and not any(x['count'] for x in list(speaker_overlap.values())+list(reference_overlap.values())),
        'identity_limits':['IDs scoped to Kathbath origin and normalized language; cross-language person identity unresolved.',
                           'Recorded parent evidence is audited locally, not re-fetched from upstream.',
                           'Expected within-split shared identities do not establish independent windows.',
                           'Shuffled-label controls do not prove absence of leakage; hashes do not rule out near-duplicates.'],
    }

"""Command line: ``tcrstruct <command> ...`` (``tcrstruct <command> -h`` for details).

    tcrstruct compare model.cif reference.pdb               one pair, printed
    tcrstruct compare-dirs predictions/ references/ -o metrics.csv -j 8
    tcrstruct compare-dirs ... --v1                         the 'v1' metrics
    tcrstruct compare-dirs ... --angles v1 --dockq v1       ... for some metrics only
    tcrstruct compare-dirs ... --angle-helices v1           Pierce lab's angles, the 'v1' helices
    tcrstruct normalize raw/ normalized/                    file or directory
    tcrstruct info structure.pdb                            chains, peptide core, CDRs, angles
    tcrstruct chains structure.pdb                          how each chain was detected
    tcrstruct verify structure.pdb | structures/            complete TCR-pMHC? chains, contacts
    tcrstruct split-peptide 6xcp.cif 6xcp_split.cif         fused peptide -> its own chain C
    tcrstruct complexes 2e7l.cif [-o dir/]                  each copy of the complex in a file
    tcrstruct split 1ao7.pdb parts/ [--parts tcr cdrs]      TCR, pMHC, MHC, peptide files
    tcrstruct rcsb-download --tcr3d II -o structures/       lists from TCR3d, files from RCSB
    tcrstruct docs [metrics]                                the documentation pages
    tcrstruct skills install [--project] [--deps]           the Claude Code skills

Warnings (chains not scored, metrics not computed, models without a reference)
are printed to stderr; ``-q`` hides them.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

__all__ = ['main']


def _number_or_none(value: str) -> int | None:
    """--complex / --assembly: a number from 1, or ``none``."""
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'a number (1, 2, ...) or none, not {value!r}')


def _complex_option(q) -> None:
    q.add_argument('--complex', type=_number_or_none, default=1, metavar='N',
                   help='for a file with several copies of the complex, which one, found by '
                        'contacts (default: 1; none: every chain, the first of each kind '
                        'by chain ID getting its standard ID)')


def _angle_method(args) -> dict:
    """The --angles method with the --angle-* overrides."""
    sign = {'signed': True, 'folded': False}.get(args.angle_sign)
    return {'method': args.angles, 'helices': args.angle_helices, 'axis': args.angle_axis,
            'signed': sign}


def _methods(args) -> dict:
    """The scoring settings: --align-on, --align-atoms, and --v1 or --angles / --dockq /
    --tm-score."""
    if args.v1:
        args.angles = 'v1'
    align_on = args.align_on[0] if len(args.align_on) == 1 else args.align_on
    return {'min_identity': args.min_identity, 'align_on': align_on,
            'align_atoms': 'heavy' if args.v1 else args.align_atoms,
            'angle_method': _angle_method(args),
            'dockq_method': 'v1' if args.v1 else args.dockq,
            'tm_method': 'v1' if args.v1 else args.tm_score}


def _save_settings(output: str, settings: dict, versions: str) -> Path:
    """Write the scoring settings and package versions next to a CSV: ``<name>.settings.json``."""
    import json
    path = Path(output).with_suffix('.settings.json')
    path.write_text(json.dumps({'versions': versions, **settings}, indent=2) + '\n')
    return path


def _compare(args):
    import pandas as pd

    from .compare import compare
    settings = _methods(args)
    row = compare(args.model, args.reference, **settings)
    table = row if isinstance(row, pd.DataFrame) else row.to_frame().T
    if args.output:
        table.to_csv(args.output, index=False)
        extra = _save_settings(args.output, settings, row.attrs['versions'])
        print(f'saved {args.output} and {extra}')
    else:
        print((row if isinstance(row, pd.Series) else table.T).to_string())
        print(row.attrs['versions'])


def _compare_dirs(args):
    from .batch import compare_dirs, key_pattern, pdb_key
    key = key_pattern(args.key_pattern) if args.key_pattern else pdb_key
    settings = _methods(args)
    table = compare_dirs(args.model_dir, args.reference_dir, n_jobs=args.jobs, key=key,
                         exclude=tuple(args.exclude), progress=True, **settings)
    table.to_csv(args.output, index=False)
    extra = _save_settings(args.output, settings, table.attrs['versions'])
    failed = (table['error'] != '').sum()
    warned = (table['warnings'] != '').sum()
    print(f'saved {len(table)} rows to {args.output} (settings in {extra})'
          + (f' ({failed} failed)' if failed else '')
          + (f'; {warned} with warnings (see the warnings column)' if warned else ''))
    for path in table.attrs.get('unmatched', []):
        print(f'  no reference: {path}', file=sys.stderr)


def _normalize(args):
    from .batch import normalize_dir
    from .chains import normalize
    kw = dict(keep_leader=args.keep_leader, heavy=args.heavy, complex=args.complex)
    src, dst = Path(args.input), Path(args.output)
    if src.is_dir():
        result = normalize_dir(src, dst, fmt=args.format, n_jobs=args.jobs, progress=True, **kw)
        failed = result[result['error'] != '']
        print(f'normalized {len(result) - len(failed)} of {len(result)} files into {dst}')
        for row in failed.itertuples():
            print(f'  failed: {row.input}: {row.error}', file=sys.stderr)
    else:
        if args.format:
            dst = dst.with_suffix(f'.{args.format}')
        normalize(src, **kw).write(dst)
        print(f'saved {dst}')


def _info(args):
    from .tcr import TCRFrame
    pf = TCRFrame.read(args.structure, complex=args.complex)
    print(pf.report())
    if pf.has_tcr:
        a = pf.docking_angles(_angle_method(args))
        print(f"  angles  crossing {a['crossing_angle']:.1f}  incident {a['incident_angle']:.1f}")


def _verify(args):
    from .verify import verify, verify_dir
    kw = dict(kind='pmhc' if args.pmhc else 'tcrpmhc', require_b2m=args.require_b2m,
              contacts=not args.no_contacts, complex=args.complex)
    if Path(args.path).is_dir():
        table = verify_dir(args.path, n_jobs=args.jobs, progress=True, **kw)
        if args.output:
            table.to_csv(args.output, index=False)
        print(f'{int(table["valid"].sum())} of {len(table)} structures pass')
        for r in table[~table['valid']].itertuples():
            print(f'  FAIL {r.file}: {r.failed}')
    else:
        table = verify(args.path, **kw)
        print(table.to_string(index=False))
        print('valid' if table.attrs['valid'] else 'NOT valid')


def _split_peptide(args):
    from .fused import split_fused_peptide
    split_fused_peptide(args.input, linker='drop' if args.drop_linker else 'keep').write(args.output)
    print(f'saved {args.output}')


def _complexes(args):
    from .complexes import complex_table, find_complexes
    if args.output:
        frames = find_complexes(args.structure)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        suffix = args.format or Path(args.structure).suffix.lstrip('.') or 'cif'
        for frame in frames:
            path = frame.write(out / f'{frame.name}.{suffix}')
            print(f'saved {path}  (file chains {", ".join(frame.metadata["complex_chains"])})')
        return
    table = complex_table(args.structure)
    print(table.drop(columns='chains').to_string())
    if table.attrs['unused']:
        print(f"chains in no complex: {', '.join(table.attrs['unused'])}")


def _split(args):
    from .split import SPLIT_PARTS
    from .tcr import TCRFrame
    pf = TCRFrame.read(args.structure, complex=args.complex)
    fmt = args.format or ('cif' if Path(args.structure).suffix in ('.cif', '.mmcif') else 'pdb')
    paths = pf.split(args.parts or list(SPLIT_PARTS), out_dir=args.output, fmt=fmt)
    for part, path in paths.items():
        print(f'{part:<10}{path}')


def _rcsb_download(args):
    from . import sources
    ids = list(args.pdb_ids)
    if args.tcr3d:
        table = sources.tcr3d_complexes(None if args.tcr3d == 'all' else args.tcr3d)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        listing = Path(args.output) / 'tcr3d_complexes.csv'
        table.to_csv(listing, index=False)
        print(f'{len(table)} TCR3d complexes listed in {listing}')
        ids += table['pdb_id'].tolist()
    if not ids:
        raise SystemExit('give PDB IDs, or --tcr3d I / II / all')
    result = sources.rcsb_download(ids, args.output, fmt=args.format, assembly=args.assembly,
                                   n_jobs=args.jobs, progress=True)
    print(result['status'].str.split(':').str[0].value_counts().to_string())


def _chains(args):
    from pdbframe import PDBFrame
    from .chains import detect_chains
    table = detect_chains(PDBFrame.read(args.structure))
    print(f'{"chain":<6}{"type":<9}{"details":<9}{"new":<5}{"length":>7}  sequence')
    for chain, row in table.iterrows():
        seq = row['sequence'] if len(row['sequence']) <= 60 else row['sequence'][:57] + '...'
        print(f'{chain:<6}{row["type"]:<9}{row["details"]:<9}{row["new_chain"]:<5}'
              f'{len(row["sequence"]):>7}  {seq}')


def _docs(args):
    from .resources import docs
    try:
        page = None if args.name is None else docs(args.name)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    if page is None:
        for name, path in docs().items():
            print(f'{name:<12}{path}')
    elif args.path:
        print(page)
    else:
        print(page.read_text())


def _skills(args):
    from .resources import install_skills, skills
    if args.action == 'list':
        for name, path in skills().items():
            print(f'{name:<26}{path}')
        return
    for name, status in install_skills(args.target, project=args.project,
                                       force=args.force, deps=args.deps).items():
        print(f'{name:<26}{status}')


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='tcrstruct', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-q', '--quiet', action='store_true', help='hide warnings')
    sub = p.add_subparsers(dest='command', required=True)

    def scoring(q):
        modes = ['peptide_centric', 'full', 'pmhc', 'mhc', 'tcr', 'v_domains', 'cdrs']
        q.add_argument('--align-on', choices=modes, nargs='+', default=['peptide_centric'],
                       help='what the superposition fits on; several give one row per '
                            'superposition (default: peptide_centric)')
        q.add_argument('--min-identity', type=float, default=0.3,
                       help='skip chains less identical to the reference (default: 0.3)')
        q.add_argument('--align-atoms', choices=['ca', 'heavy'], default='ca',
                       help="atoms the superposition is fitted on: the C-alpha ('ca') or all "
                            "heavy atoms of the --align-on region (default: ca)")
        q.add_argument('--dockq', choices=['v2', 'v1'], default='v2',
                       help="'v2': DockQ v2, TCR against pMHC; 'v1': peptide core "
                            "against MHC and CDRs, no refit (default: v2)")
        q.add_argument('--tm-score', choices=['v2', 'v1'], default='v2',
                       help="'v2': all matched C-alpha atoms on their own fit; "
                            "'v1': in the superposition's pose, no fit of its own (default: v2)")
        q.add_argument('--v1', action='store_true',
                       help="the 'v1' metrics: --align-atoms heavy --angles v1 "
                            "--dockq v1 --tm-score v1")
        angles(q)

    def angles(q):
        q.add_argument('--angles', choices=['piercelab', 'v1'], default='piercelab',
                       help="docking angles: 'piercelab' (Pierce's helix residues, V-domain axis, "
                            "0-180) or 'v1' (Motif Atlas helices, whole-chain axis, 0-90) "
                            "(default: piercelab)")
        q.add_argument('--angle-helices', choices=['imgt', 'v1'],
                       help='override the helix residues of --angles')
        q.add_argument('--angle-axis', choices=['piercelab', 'v1'],
                       help='override the rotation axis of --angles (V domains / all IMGT labels)')
        q.add_argument('--angle-sign', choices=['signed', 'folded'],
                       help='override the signs of --angles (0-180 / 0-90)')

    q = sub.add_parser('compare', help='score one model against its reference')
    q.add_argument('model')
    q.add_argument('reference')
    q.add_argument('-o', '--output', help='CSV file (default: print)')
    scoring(q)
    q.set_defaults(func=_compare)

    q = sub.add_parser('compare-dirs', help='score every model in a directory',
                       description='Models pair with references by the first 4 letters of '
                                   'the file name (7N2N_model_0.pdb -> 7N2N.pdb), recursively.')
    q.add_argument('model_dir')
    q.add_argument('reference_dir')
    q.add_argument('-o', '--output', required=True, help='CSV file')
    q.add_argument('-j', '--jobs', type=int, default=1, help='processes (-1: all CPUs)')
    q.add_argument('--key-pattern', help='regex whose first group is the PDB ID in a file name')
    q.add_argument('--exclude', nargs='*', default=[],
                   help='skip model sub-directories with these names (e.g. rcsb)')
    scoring(q)
    q.set_defaults(func=_compare_dirs)

    q = sub.add_parser('normalize', help='standard chain IDs and IMGT-numbered TCRs',
                       description='Rename chains to A (MHC), B (MHC beta), C (peptide), '
                                   'D (TCR alpha), E (TCR beta), F (B2M) and renumber the '
                                   'TCR chains by IMGT.')
    q.add_argument('input', help='structure file or directory')
    q.add_argument('output', help='output file or directory')
    q.add_argument('--format', choices=['pdb', 'cif'], help='output format (default: as input)')
    q.add_argument('--keep-leader', action='store_true',
                   help='keep TCR residues before the V domain (numbered -k..-1)')
    q.add_argument('--heavy', action='store_true', help='drop hydrogens')
    _complex_option(q)
    q.add_argument('-j', '--jobs', type=int, default=1, help='processes, for a directory')
    q.set_defaults(func=_normalize)

    q = sub.add_parser('info', help='chains, peptide core, CDRs and docking angles')
    q.add_argument('structure')
    _complex_option(q)
    angles(q)
    q.set_defaults(func=_info)

    q = sub.add_parser('verify', help='check a structure (or a directory) is a complete complex')
    q.add_argument('path', help='structure file or directory')
    q.add_argument('--pmhc', action='store_true', help='no TCR expected')
    q.add_argument('--require-b2m', action='store_true',
                   help='class I without B2M is a failure (by default a warning)')
    q.add_argument('--no-contacts', action='store_true', help='skip the contact checks')
    _complex_option(q)
    q.add_argument('-o', '--output', help='CSV of the results, for a directory')
    q.add_argument('-j', '--jobs', type=int, default=1, help='processes, for a directory')
    q.set_defaults(func=_verify)

    q = sub.add_parser('split-peptide', help='move a peptide fused to another chain into chain C')
    q.add_argument('input')
    q.add_argument('output')
    q.add_argument('--drop-linker', action='store_true', help='remove the linker residues')
    q.set_defaults(func=_split_peptide)

    q = sub.add_parser('complexes', help='find each TCR-pMHC complex in a multi-copy file',
                       description='Chains grouped by type and contacts (C-alpha within 8 A); '
                                   'prints which file chains make up each complex, or writes '
                                   'each as its own file with -o.')
    q.add_argument('structure')
    q.add_argument('-o', '--output', help='directory to write <name>_<n> files to')
    q.add_argument('--format', choices=['pdb', 'cif'], help='output format (default: as input)')
    q.set_defaults(func=_complexes)

    q = sub.add_parser('split', help='write the TCR, pMHC, MHC and peptide as their own files')
    q.add_argument('structure')
    q.add_argument('output', help='directory')
    q.add_argument('--parts', nargs='*',
                   help='parts or regions (default: tcr pmhc mhc peptide; also e.g. cdrs, '
                        'cdr3, peptide_core)')
    _complex_option(q)
    q.add_argument('--format', choices=['pdb', 'cif'], help='output format (default: as input)')
    q.set_defaults(func=_split)

    q = sub.add_parser('rcsb-download', help='download structures (or FASTA) from the RCSB PDB')
    q.add_argument('pdb_ids', nargs='*', help='PDB IDs')
    q.add_argument('--tcr3d', choices=['I', 'II', 'all'],
                   help="also every TCR-pMHC in TCR3d of this class (saves the list too)")
    q.add_argument('-o', '--output', required=True, help='directory')
    q.add_argument('--format', choices=['cif', 'pdb', 'fasta'], default='cif')
    q.add_argument('--assembly', type=_number_or_none, default=1, metavar='N',
                   help='biological assembly (default: 1; none: the deposited structure)')
    q.add_argument('-j', '--jobs', type=int, default=4)
    q.set_defaults(func=_rcsb_download)

    q = sub.add_parser('chains', help='how each chain of a structure was detected')
    q.add_argument('structure')
    q.set_defaults(func=_chains)

    q = sub.add_parser('docs', help='the documentation pages (tcrstruct and PDBFrame)',
                       description='Without a name, lists the pages and their files; with '
                                   'one, prints it.')
    q.add_argument('name', nargs='?', help='a page, e.g. metrics, tcrframe, datasets, cli')
    q.add_argument('--path', action='store_true', help="print the page's file, not its text")
    q.set_defaults(func=_docs)

    q = sub.add_parser('skills', help='the Claude Code skills for this package',
                       description='install copies them into ~/.claude/skills (every '
                                   'project) or, with --project, ./.claude/skills.')
    q.add_argument('action', choices=['install', 'list'])
    q.add_argument('--project', action='store_true', help='into ./.claude/skills')
    q.add_argument('--target', help='another skills folder to copy into')
    q.add_argument('--force', action='store_true', help='replace skills that differ')
    q.add_argument('--deps', action='store_true', help="also PDBFrame's and tcrkit's skills")
    q.set_defaults(func=_skills)
    return p


def _show(message, category, filename, lineno, file=None, line=None):
    print(f'warning: {message}', file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with warnings.catch_warnings():
        if args.quiet:
            warnings.simplefilter('ignore')
        else:
            warnings.simplefilter('always')
            warnings.showwarning = _show
        args.func(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())

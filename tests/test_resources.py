"""The documentation and skills that come with the package: tcs.docs, install_skills, CLI."""

import tomllib
from pathlib import Path

import pytest

import tcrstruct as tcs
from tcrstruct.cli import main
from tcrstruct.resources import SKILLS, skills

ROOT = Path(__file__).parents[1]


def test_docs():
    pages = tcs.docs()
    assert {'tcrframe', 'metrics', 'datasets', 'cli'} <= set(pages)
    assert 'README' not in pages
    assert tcs.docs('metrics.md') == tcs.docs('metrics') == pages['metrics']
    assert tcs.docs('metrics').read_text().startswith('# Metrics')
    with pytest.raises(FileNotFoundError, match='the pages are .*metrics'):
        tcs.docs('nowhere')
    for dep, page in [('pdbframe', 'seqframe'), ('tcrkit', 'api')]:
        if hasattr(__import__(dep), 'docs'):       # versions that ship their docs
            assert tcs.docs(page) == __import__(dep).docs(page)


def test_skills_match_the_wheel():
    found = skills()
    assert list(found) == SKILLS
    for name, folder in found.items():
        assert f'name: {name}\n' in (folder / 'SKILL.md').read_text()
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    packaged = config['tool']['hatch']['build']['targets']['wheel']['force-include']
    assert packaged.pop('docs') == 'tcrstruct/docs'
    assert packaged == {f'.claude/skills/{s}': f'tcrstruct/skills/{s}' for s in SKILLS}


def test_install_skills(tmp_path):
    assert set(tcs.install_skills(tmp_path).values()) == {'installed'}
    assert (tmp_path / 'score-tcrpmhc-models' / 'SKILL.md').is_file()
    assert set(tcs.install_skills(tmp_path).values()) == {'up to date'}
    edited = tmp_path / 'show-tcrpmhc' / 'SKILL.md'
    edited.write_text('my own version')
    assert tcs.install_skills(tmp_path)['show-tcrpmhc'].startswith('kept')
    assert edited.read_text() == 'my own version'
    assert tcs.install_skills(tmp_path, force=True)['show-tcrpmhc'] == 'updated'
    assert edited.read_text() == (skills()['show-tcrpmhc'] / 'SKILL.md').read_text()
    with_deps = tcs.install_skills(tmp_path / 'all', deps=True)
    assert set(SKILLS) < set(with_deps)
    for dep in ('pdbframe', 'tcrkit'):
        module = __import__(dep)
        if hasattr(module, 'install_skills'):           # their skills, installed too
            assert set(module.resources.SKILLS) <= set(with_deps)
        else:                                           # an older version: what to do
            assert with_deps[dep].endswith('update it')


def test_cli(tmp_path, capsys):
    assert main(['docs']) == 0
    assert 'metrics' in capsys.readouterr().out
    assert main(['docs', 'cli']) == 0
    assert capsys.readouterr().out.startswith('# Command line')
    assert main(['docs', 'cli', '--path']) == 0
    assert capsys.readouterr().out.strip() == str(tcs.docs('cli'))
    with pytest.raises(SystemExit, match='no page'):
        main(['docs', 'nowhere'])
    assert main(['skills', 'install', '--target', str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(SKILLS)

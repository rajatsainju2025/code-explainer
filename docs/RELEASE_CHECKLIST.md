# Release Checklist

- [ ] Update version in src/code_explainer/__init__.py
- [ ] Update version in setup.py / pyproject.toml
- [ ] Update CHANGELOG.md
- [ ] Run tests: `pytest`
- [ ] Build: `python -m build`
- [ ] Publish (TestPyPI): `twine upload -r testpypi dist/*`
- [ ] Publish (PyPI): `twine upload dist/*`
- [ ] Create GitHub Release with notes and tag

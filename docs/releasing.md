# Release procedure

MantiShrimp releases are published from GitHub. A published GitHub Release
triggers the PyPI Trusted Publishing workflow and, once the repository is
enabled in Zenodo, its software archive.

## One-time account setup

1. Create and verify a PyPI account with two-factor authentication.
2. In the GitHub repository settings, create an environment named `pypi` and
   require a maintainer's approval for deployment.
3. In the PyPI account's **Publishing** page, add a pending GitHub publisher:

   - project: `mantishrimp`
   - owner: `sthsci`
   - repository: `MantiShrimp`
   - workflow: `release.yml`
   - environment: `pypi`

4. Sign into Zenodo with GitHub, open the GitHub integration, click **Sync
   now**, and enable `sthsci/MantiShrimp`.

Complete these steps before publishing the first GitHub Release. A pending
PyPI publisher does not reserve the project name until the first successful
upload.

## Release checklist

1. Update the version in `pyproject.toml`, `src/mantishrimp/__init__.py`, and
   `CITATION.cff`.
2. Move the relevant entries from `CHANGELOG.md` under the new version and set
   its release date.
3. Confirm author, ORCID, affiliation, and contribution metadata.
4. Run:

   ```bash
   python -m pip install -e '.[all,test]'
   python -m pytest
   python -m build
   python -m twine check --strict dist/*
   ```

5. Push the release-preparation commit and wait for the full CI matrix.
6. On GitHub, create a release from `main` with a new tag matching the package
   version, for example `v0.1.0`. Use the changelog entry as release notes.
7. Publish the GitHub Release. Do not create the tag separately: GitHub creates
   it from the selected release target.
8. Approve the `pypi` deployment when the release workflow requests it.
9. Verify the PyPI page and a clean installation of the base and inference
   extras.
10. Wait for Zenodo to finish archiving and check the creator metadata and
    version DOI. Add the resulting DOI badge to the README in a follow-up
    commit.

PyPI release files and Zenodo software versions are persistent. If code or
package metadata must change after publication, increment the package version
instead of trying to replace `0.1.0`.

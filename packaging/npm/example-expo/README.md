# Expo release CI fixture

This is a release CI fixture for validating the assembled npm package against
Expo native builds. The checked-in `file:..` dependency points at the package
template only; release CI replaces it with the assembled package tarball before
installing dependencies.

For application setup and published-package usage, follow the parent package
README instead of running this fixture directly from a clean checkout.

"""Task registrations for tracking-bfm."""

from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp", ".scripts"]

import_packages(__name__, _BLACKLIST_PKGS)

"""Every command Compass exposes, one module apiece.

Importing this package imports all of them, and importing a command module is
what attaches its command to the ``cli`` group in :mod:`compass.cli.app`. The
modules are imported for that side effect alone — nothing here is re-exported.
"""

from compass.cli.commands import (  # noqa: F401
    analyze,
    checkpoint,
    compare,
    docs,
    grade,
    imports,
    init,
    insights,
    listing,
    run,
    site,
    trace,
)

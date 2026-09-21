#!/bin/bash
#
# Make the container's user match whoever owns the mounted notebooks directory.
#
# The base image runs as `jovyan`, uid 1000. A bind mount keeps the host's ownership, so
# on any machine whose user is not uid 1000 the container cannot write to it — students
# cannot even save the notebook they are working in. Docker Desktop hides this on macOS
# and Windows, which is why it tends to surface only on Linux, mid-class.
#
# Rather than force the host's files to uid 1000, we adopt the host's uid instead. The
# base image's start.sh already knows how to rebuild the user from NB_UID/NB_GID when the
# container starts as root; we only have to work out what those values should be.
set -euo pipefail

WORK_DIR="/home/jovyan/work"

if [ -d "${WORK_DIR}" ]; then
    owner_uid="$(stat -c '%u' "${WORK_DIR}")"
    owner_gid="$(stat -c '%g' "${WORK_DIR}")"

    # uid 0 means the mount is root-owned, which happens when the directory did not exist
    # on the host and Docker created it. Running the notebook as root is worse than the
    # permission problem, so leave the default user alone in that case.
    if [ "${owner_uid}" != "0" ]; then
        export NB_UID="${owner_uid}"
        export NB_GID="${owner_gid}"
        export CHOWN_HOME="yes"
        export CHOWN_HOME_OPTS="-R"
        echo "[entrypoint] adopting host ownership of ${WORK_DIR}: uid=${NB_UID} gid=${NB_GID}"
    fi
fi

exec /usr/local/bin/start.sh "$@"

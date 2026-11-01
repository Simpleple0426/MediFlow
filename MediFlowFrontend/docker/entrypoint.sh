#!/bin/sh
set -eu

cat >/usr/share/nginx/html/runtime-config.js <<EOF
window.__MEDIFLOW_CONFIG__ = {
  enabledBackends: "${ENABLED_BACKENDS:-python,java}",
  pythonApiUrl: "${PYTHON_API_URL:-/api/python}",
  javaApiUrl: "${JAVA_API_URL:-/api/java}"
};
EOF

exec nginx -g "daemon off;"

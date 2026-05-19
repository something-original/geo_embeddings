#!/bin/sh
set -e

_read_secret() {
  var_name="$1"
  file_path="$2"
  if [ -f "$file_path" ]; then
    value=$(cat "$file_path" | tr -d '\r')
    if [ -n "$value" ]; then
      export "$var_name=$value"
    fi
  fi
}

_read_secret DB_PWD /run/secrets/db_password
_read_secret HF_TOKEN /run/secrets/hf_token
_read_secret QDRANT_API_KEY /run/secrets/qdrant_api_key

exec "$@"

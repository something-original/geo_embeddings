#!/bin/sh
# Read sensitive keys from .env and write Docker Compose secret files.
set -e
cd "$(dirname "$0")/.."

ENV_FILE="${1:-.env}"
SECRETS_DIR="secrets"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE — copy .env.example to .env first." >&2
  exit 1
fi

mkdir -p "$SECRETS_DIR"

# shellcheck disable=SC1090
. "$ENV_FILE"

write_secret() {
  name="$1"
  value="$2"
  file="$SECRETS_DIR/$name"
  if [ -z "$value" ]; then
    : > "$file"
    echo "  $name (empty)"
  else
    printf '%s' "$value" > "$file"
    echo "  $name"
  fi
  chmod 600 "$file"
}

echo "Writing secrets to $SECRETS_DIR/"
write_secret db_password "${DB_PWD:-}"
write_secret hf_token "${HF_TOKEN:-}"
write_secret qdrant_api_key "${QDRANT_API_KEY:-}"
echo "Done."

#!/usr/bin/env bash
# Remove preftool from this machine.
# Run `preftool uninstall` inside each study repo FIRST - this script only
# removes the tool itself, not the changes it made to your repos.
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"

echo "Removing the PATH entry added by install.sh..."
for RC in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile"; do
  [ -f "$RC" ] || continue
  grep -q "preftool PATH" "$RC" || continue
  # Drop our comment line, the export line under it, and the blank line
  # install.sh put in front - buffer blanks so they only survive if real
  # content follows them.
  awk '
    /^$/                                            { blank = blank "\n"; next }
    /^# preftool PATH \(added by install\.sh\)$/   { skip = 1; blank = ""; next }
    skip == 1                                       { skip = 0; next }
                                                    { printf "%s", blank; blank = ""; print }
  ' "$RC" > "$RC.preftool-tmp" && mv "$RC.preftool-tmp" "$RC"
  echo "  cleaned $RC"
done

rm -rf .venv
echo "  removed $HERE/.venv"

echo
echo "Done. The only thing left is this folder:"
echo
echo "    rm -rf \"$HERE\""
echo
echo "Note: your Claude Code transcripts in ~/.claude/ are yours and were not"
echo "touched. If you installed Entire for this study and no longer want it:"
echo "    brew uninstall --cask entire && brew untap entireio/tap"

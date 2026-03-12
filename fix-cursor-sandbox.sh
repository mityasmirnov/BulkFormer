#!/bin/bash
# Fix Cursor terminal sandbox on Ubuntu 24.04+ (AppArmor 4.x)
# Run with: sudo bash fix-cursor-sandbox.sh

set -e

PROFILE="/etc/apparmor.d/cursor-sandbox-remote"
BACKUP="$PROFILE.bak.$(date +%Y%m%d%H%M%S)"

echo "=== Cursor sandbox AppArmor fix ==="

# Backup
cp -a "$PROFILE" "$BACKUP"
echo "Backed up to $BACKUP"

# Write patched profile with all required rules
cat > "$PROFILE" << 'EOF'
abi <abi/4.0>,
include <tunables/global>

profile cursor_sandbox_remote /home/*/.cursor-server/bin/*/*/resources/helpers/cursorsandbox {
  file,
  /** ix,

  capability sys_admin,
  capability net_admin,
  capability chown,
  capability setuid,
  capability setgid,
  capability setpcap,
  capability dac_override,

  userns,

  network unix stream,
  network unix dgram,
  network netlink raw,
  network netlink dgram,

  mount,
  remount,
  umount,

  /home/*/.cursor-server/bin/*/*/resources/helpers/cursorsandbox mr,
}

profile cursor_sandbox_agent_cli /home/*/.local/share/cursor-agent/versions/*/cursorsandbox {
  file,
  /** ix,

  capability sys_admin,
  capability net_admin,
  capability chown,
  capability setuid,
  capability setgid,
  capability setpcap,
  capability dac_override,

  userns,

  network unix stream,
  network unix dgram,
  network netlink raw,
  network netlink dgram,

  mount,
  remount,
  umount,

  /home/*/.local/share/cursor-agent/versions/*/cursorsandbox mr,
}
EOF

echo "Reloading AppArmor..."
apparmor_parser -r "$PROFILE"

echo ""
echo "Done. Restart Cursor and try the Agent terminal again."
echo ""
echo "If it still fails, try the sysctl workaround:"
echo "  echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/99-cursor.conf"
echo "  sudo sysctl --system"
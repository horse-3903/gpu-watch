#!/usr/bin/env bash
set -euo pipefail

PKG_NAME="gpu-watch"
VERSION="0.1.0"
ARCH="all"
PKG_FULL="${PKG_NAME}_${VERSION}_${ARCH}"
BUILD_ROOT="build/${PKG_FULL}"
DIST_DIR="dist"

echo "[build_deb] Building ${PKG_FULL}.deb ..."

# Clean previous build
rm -rf "$BUILD_ROOT"
mkdir -p "${BUILD_ROOT}/DEBIAN"
mkdir -p "${BUILD_ROOT}/usr/local/bin"

# Copy scripts
cp src/gpu-watch                 "${BUILD_ROOT}/usr/local/bin/gpu-watch"
cp src/gpu_watch_metrics.py      "${BUILD_ROOT}/usr/local/bin/gpu_watch_metrics.py"

# Set permissions
chmod 755 "${BUILD_ROOT}/usr/local/bin/gpu-watch"
chmod 755 "${BUILD_ROOT}/usr/local/bin/gpu_watch_metrics.py"

# Write control file
cat > "${BUILD_ROOT}/DEBIAN/control" <<EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: gpu-watch
Depends: bash, curl, python3, coreutils, procps
Description: Reusable GPU pod training watcher with logs, ntfy alerts, GPU stats, and generic metrics
 gpu-watch wraps any training or evaluation command and provides:
 live terminal output, persistent timestamped logs, ntfy.sh push
 notifications, GPU monitoring via nvidia-smi, and generic model
 metric extraction from TensorBoard, CSV, JSON, or stdout.
EOF

# Write postinst
cat > "${BUILD_ROOT}/DEBIAN/postinst" <<'EOF'
#!/bin/bash
set -e
chmod 755 /usr/local/bin/gpu-watch
chmod 755 /usr/local/bin/gpu_watch_metrics.py
EOF
chmod 755 "${BUILD_ROOT}/DEBIAN/postinst"

# Write prerm
cat > "${BUILD_ROOT}/DEBIAN/prerm" <<'EOF'
#!/bin/bash
set -e
EOF
chmod 755 "${BUILD_ROOT}/DEBIAN/prerm"

# Build .deb
mkdir -p "$DIST_DIR"
dpkg-deb --build "$BUILD_ROOT" "${DIST_DIR}/${PKG_FULL}.deb"

echo "[build_deb] Done: ${DIST_DIR}/${PKG_FULL}.deb"

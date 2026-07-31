# Bump this any time the C++ templates or runtime headers change.
# It is embedded in the SimulationSpec payload and therefore in the build-cache key,
# so incrementing it invalidates all cached extensions and forces a fresh compile.
BACKEND_VERSION = "0.3"

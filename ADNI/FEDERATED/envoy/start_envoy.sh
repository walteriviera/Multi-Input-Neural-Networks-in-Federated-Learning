#!/bin/bash
set -e
ENVOY_NAME=$1
ENVOY_CONF=$2
DIRECTOR=$3
fx envoy start -n "$ENVOY_NAME" --disable-tls --envoy-config-path "$ENVOY_CONF" -dh "$DIRECTOR" -dp 50051

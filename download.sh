#!/usr/bin/env bash
set -eu

# Download some datasets from:
# https://homepages.cwi.nl/~aeb/go/games/index.html/

ROOT=https://homepages.cwi.nl/~aeb/go/games
NAMES="dosaku shusaku shusai takagawa go_seigen alphago"

# Create data directory if it doesn't exist
if [ ! -d data ]; then
   mkdir data
fi

# Download the sgfs
pushd data

for name in $NAMES; do
    file="${ROOT}/${name}.tgz"
    wget $file --no-check-certificate
done

for tgz in *.tgz; do
    tar -xvzf $tgz
    rm $tgz
done

popd data

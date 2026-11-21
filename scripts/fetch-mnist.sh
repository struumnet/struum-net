#!/usr/bin/env bash
# Downloads the MNIST handwritten digits (IDX format) into data/mnist/.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/data/mnist"
MIRROR="${MNIST_MIRROR:-https://ossci-datasets.s3.amazonaws.com/mnist}"
mkdir -p "$DIR"
cd "$DIR"

# name md5 (from torchvision's MNIST dataset definition)
while read -r name md5; do
  if [ ! -f "$name" ]; then
    echo "Downloading $name.gz"
    curl -fsSL "$MIRROR/$name.gz" -o "$name.gz"
    echo "$md5  $name.gz" | md5sum -c --quiet
    gzip -d "$name.gz"
  fi
done <<'LIST'
train-images-idx3-ubyte f68b3c2dcbeaaa9fbdd348bbdeb94873
train-labels-idx1-ubyte d53e105ee54ea40749a09fcbcd1e9432
t10k-images-idx3-ubyte 9fb629c4189551a2d022fa330f9573f3
t10k-labels-idx1-ubyte ec29112dd5afa0611ce80d1b7f02629c
LIST
echo "MNIST ready in $DIR"

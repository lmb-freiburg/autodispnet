#!/bin/bash
URL_BASE="https://lmb.informatik.uni-freiburg.de/resources/binaries/autodispnet/nets"

download () {
	net=$1
	evo=$2
	state=$3
	subpath="$net/training/$evo/checkpoints"
	wget --no-check-certificate "$URL_BASE/$subpath/snapshot-$state.data-00000-of-00001" -P $subpath
	wget --no-check-certificate "$URL_BASE/$subpath/snapshot-$state.index" -P $subpath
	wget --no-check-certificate "$URL_BASE/$subpath/snapshot-$state.meta" -P $subpath
}

download CSS 00__FT3D__genotype 150000
download css 00__FT3D__genotype 600000
download CSS-kitti 00__kitti__genotype 90000

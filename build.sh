#!/bin/bash

yellow=`tput setaf 3`
reset_color=`tput sgr0`

ARCH="$(uname -m)"

main () {
    docker build . \
        -t dummy_triton 
    popd;
}

main "$@"; exit;
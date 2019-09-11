#!/bin/bash

for d in `find . -type d`; do
    if [ -f "${d}/rebuild-host.sh" ]; then
        echo "#################### Rebuilding ${d} ####################"
        cd "$d"
        ./rebuild-host.sh
        ./rebuild-platform.sh --staging
        cd ..
    fi
done

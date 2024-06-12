#!/bin/bash
# usage: rebuild-all.sh [a33|pro|all]

if [ "X$1" = "Xa33" ]; then build_a33=1
elif [ "X$1" = "Xpro" ]; then build_pro=1
elif [ "X$1" = "Xall" ]; then build_a33=1; build_pro=1;
else echo "usage: rebuild-all.sh [a33|pro|all]"; exit 2;
fi

for d in `find . -type d`; do
    if [ -f "${d}/rebuild-host.sh" ]; then
        echo "#################### Rebuilding ${d} ####################"
        cd "$d"
        if [ "X$build_a33" = "X1" ]; then
            ./rebuild-host.sh
            ./rebuild-platform.sh --staging
        fi
        
        if [ "X$build_pro" = "X1" ]; then
            ./rebuild-pro-host.sh
            ./rebuild-pro-platform-pdeb.sh
        fi
        cd ..
    fi
done

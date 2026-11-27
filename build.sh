#!/usr/bin/env sh
# Build, test or run the Gram-Schmidt program. Needs JDK 17 or newer.
#
#   ./build.sh            compile into build/classes
#   ./build.sh test       compile and run the unit tests
#   ./build.sh run ...    compile, then run with the given arguments
#   ./build.sh clean      remove the build directory
set -eu
cd "$(dirname "$0")"

compile() {
    rm -rf build
    mkdir -p build/classes build/test-classes
    javac -encoding UTF-8 -Xlint:all -Werror -d build/classes src/GramSchmidt.java
}

case "${1:-build}" in
    build)
        compile
        echo "Built - run it with: java -cp build/classes GramSchmidt --demo"
        ;;
    test)
        compile
        javac -encoding UTF-8 -Xlint:all -Werror -cp build/classes -d build/test-classes test/GramSchmidtTest.java
        java -cp build/classes:build/test-classes GramSchmidtTest
        ;;
    run)
        shift
        compile
        java -cp build/classes GramSchmidt "$@"
        ;;
    clean)
        rm -rf build
        ;;
    *)
        echo "usage: $0 [build|test|run [args...]|clean]" >&2
        exit 2
        ;;
esac

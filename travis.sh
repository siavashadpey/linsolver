#!/bin/bash

keyword=$1

set -o pipefail

if [ $keyword == build ]; then
	mkdir build_tests
	cd build_tests || exit
	cmake -DBUILD_TESTS=ON -DWITH_MEMCHECK=ON ..
	make
	cd ..
elif [ $keyword == run ]; then
	cd build_tests || exit
	ctest
	ctest -T memcheck
	cd ..
#elif [ $keyword == coverage ]; then
#	gcovr -r .. "src"  --html-details -o cov.html
else
	echo "invalid keyword."
fi
#!/bin/bash

keyword=$1


if [ $keyword == build ]; then
	mkdir build_tests
	cd build_tests || exit
	cmake -DBUILD_TESTS=ON -DWITH_MEMCHECK=ON ..
	make
	cd ..
elif [ $keyword == run ]; then
	cd build_tests || exit
	ctest
	ctest --overwrite MemoryCheckCommandOptions="--leak-check=full --show-reachable=yes --error-exitcode=100" -T memcheck
	cd ..
#elif [ $keyword == coverage ]; then
#	gcovr -r .. "src"  --html-details -o cov.html
else
	echo "invalid keyword."
fi
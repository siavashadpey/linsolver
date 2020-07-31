#!/bin/bash

keyword=$1


if [ $keyword == build ]; then
	mkdir build_tests
	cd build_tests
	cmake -DBUILD_TESTS=ON ..
	make
	cd ..
elif [ $keyword == run ]; then
	cd build_tests
	ctest
#elif [ $keyword == coverage ]; then
#	gcovr -r .. "src"  --html-details -o cov.html
else
	echo "invalid keyword."
fi
#!/bin/bash
set -x # echo on
mkdir -p bin
clang++ main.mm -mmacosx-version-min=10.13 -std=c++17 -framework Metal -framework IOKit -fobjc-arc -o bin/depth_stencil_tests

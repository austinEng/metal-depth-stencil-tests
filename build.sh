#!/bin/bash
set -x # echo on
mkdir -p bin
clang++ stencil_tests.mm -mmacosx-version-min=10.12 -std=c++17 -framework Metal -fobjc-arc -o bin/stencil_tests

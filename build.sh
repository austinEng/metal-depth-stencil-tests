#!/bin/bash
set -x # echo on
mkdir -p bin
clang++ stencil_tests.mm -std=c++17 -framework Metal -fobjc-arc -o bin/stencil_tests

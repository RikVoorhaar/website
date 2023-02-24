#!/bin/bash

# This script uses perl to replace any single dollar sign with two dollar signs.

perl -pi -e 's/(?<!\$)\$(?!\$)/\$\$/g' "$1"
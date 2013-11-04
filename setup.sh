#!/bin/sh

export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.7.0_25.jdk/Contents/Home/
export JUNTO_DIR="$PWD/lib/junto"
export PATH="$PATH:$JUNTO_DIR/bin"

cd $JUNTO_DIR
bin/build update compile
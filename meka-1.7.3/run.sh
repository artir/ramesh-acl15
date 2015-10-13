#!/bin/bash

MEMORY=1024m
MAIN=meka.gui.explorer.Explorer

java -Xmx$MEMORY -cp "./lib/*" $MAIN $1


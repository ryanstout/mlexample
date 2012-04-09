#!/bin/sh

# java -server -Xmx3000m -jar `dirname $0`/weka-3-6-4/weka.jar

export WEKA_HOME=`pwd`/experiments/weka/weka-3-7-5/packages

java -server -Xmx4000m -jar `dirname $0`/weka-3-7-5/weka.jar


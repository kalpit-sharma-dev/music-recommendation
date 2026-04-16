#!/bin/bash

# generate cluster id if not already formatted
if [ ! -f /tmp/kafka-formatted ]; then
    export CLUSTER_ID=$(kafka-storage.sh random-uuid)
    kafka-storage.sh format -t $CLUSTER_ID -c /etc/kafka/kraft/server.properties
    touch /tmp/kafka-formatted
fi

exec kafka-server-start.sh /etc/kafka/kraft/server.properties

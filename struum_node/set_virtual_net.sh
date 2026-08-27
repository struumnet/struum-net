#!/bin/bash
set -e

cleanup() {
    sudo ip netns del node1 2>/dev/null || true
    sudo ip netns del node2 2>/dev/null || true
    sudo ip link del br0 2>/dev/null || true
}

if [ "$1" == "clean" ]; then
    cleanup
    exit 0
fi

cleanup

sudo ip netns add node1
sudo ip netns add node2

sudo ip link add veth1 type veth peer name eth1
sudo ip link add veth2 type veth peer name eth2

sudo ip link set eth1 netns node1
sudo ip link set eth2 netns node2

sudo ip link add br0 type bridge
sudo ip link set br0 up
sudo ip link set veth1 master br0
sudo ip link set veth2 master br0
sudo ip link set veth1 up
sudo ip link set veth2 up

sudo ip netns exec node1 ip link set lo up
sudo ip netns exec node1 ip link set eth1 up
sudo ip netns exec node1 ip addr add 10.0.0.1/24 dev eth1

sudo ip netns exec node2 ip link set lo up
sudo ip netns exec node2 ip link set eth2 up
sudo ip netns exec node2 ip addr add 10.0.0.2/24 dev eth2

echo "=== node1 ==="
sudo ip netns exec node1 ip addr
sudo ip netns exec node1 ip route

echo "=== node2 ==="
sudo ip netns exec node2 ip addr
sudo ip netns exec node2 ip route

echo "=== connectivity check ==="
sudo ip netns exec node1 ping -c 2 -W 1 10.0.0.2

#!/bin/bash
# Install nexmon CSI tool for CM5 (BCM43455 chipset)
# Gives raw per-subcarrier CSI from the built-in WiFi

set -e
echo "Installing nexmon CSI driver for CM5..."

sudo apt install -y git raspberrypi-kernel-headers build-essential cmake libgmp3-dev gawk qpdf bison flex make libtool-bin

cd /tmp
git clone https://github.com/nexmonster/nexmon_csi.git
cd nexmon_csi

# Build for CM5
export NEXMON_ROOT=$(pwd)
source setup_env.sh
make

sudo make install
sudo modprobe brcmfmac
sudo nexutil -Iwlan0 -s500 -b -l34 -v2dd0

echo "nexmon CSI installed. Interface: wlan0"
echo "Read CSI with: sudo tcpdump -i wlan0 dst port 5500 -vv -w csi.pcap"

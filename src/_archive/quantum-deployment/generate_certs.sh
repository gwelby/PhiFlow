#!/bin/bash
openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
  -keyout quantum.key -out quantum.crt -subj "/CN=demoguru.networkinggurus.com" \
  -addext "subjectAltName=DNS:demoguru.networkinggurus.com,DNS:*.demoguru.networkinggurus.com,IP:192.168.100.32"
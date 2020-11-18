#!/bin/bash

rsync -avPu mirror_api --exclude .git --exclude __pycache__ --exclude .log --exclude .idea lab@192.168.20.4:/home/lab/Appdata/mirror_v2 &
rsync -avPu mirror_api --exclude .git --exclude __pycache__ --exclude .log --exclude .idea lab@192.168.20.5:/home/lab/mirror_v2 &
rsync -avPu mirror_api --exclude .git --exclude __pycache__ --exclude .log --exclude .idea ops@192.168.100.2:mirror_v2 &
rsync -avPu mirror_api --exclude .git --exclude __pycache__ --exclude .log --exclude .idea luogang@192.168.40.131:Local/ &

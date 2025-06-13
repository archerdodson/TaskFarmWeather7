#!/bin/bash

git add .

read -p "Enter text of the commit: " commit_input

git commit -m "$commit_input"
git push origin cluster

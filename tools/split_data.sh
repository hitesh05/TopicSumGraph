#!/bin/bash

python datasplitter.py --input_file /scratch/hitesh.goel/cache/CNNDM/train.w2s.tfidf.jsonl --num_files 2000 --output_dir /scratch/hitesh.goel/cache/CNNDM/train --index_to_file_mapping /scratch/hitesh.goel/cache/CNNDM/index_to_file_mapping_train.json
python datasplitter.py --input_file /scratch/hitesh.goel/cache/CNNDM/val.w2s.tfidf.jsonl --num_files 2000 --output_dir /scratch/hitesh.goel/cache/CNNDM/val --index_to_file_mapping /scratch/hitesh.goel/cache/CNNDM/index_to_file_mapping_val.json
python datasplitter.py --input_file /scratch/hitesh.goel/cnndm/val.label.jsonl --num_files 2000 --output_dir /scratch/hitesh.goel/cnndm/val --index_to_file_mapping /scratch/hitesh.goel/cnndm/index_to_file_mapping_val.json
python datasplitter.py --input_file /scratch/hitesh.goel/cnndm/train.label.jsonl --num_files 2000 --output_dir /scratch/hitesh.goel/cnndm/train --index_to_file_mapping /scratch/hitesh.goel/cnndm/index_to_file_mapping_train.json